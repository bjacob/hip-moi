#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode ROCprof ATT output and validate ping-pong instruction scheduling."""

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


STATUS_SUCCESS = 0
STATUS_ERROR_OUT_OF_RESOURCES = 2
STATUS_ERROR_INVALID_ARGUMENT = 3
RECORD_WAVE = 3
RECORD_INFO = 4


class PcInfo(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("code_object_id", ctypes.c_uint64),
    ]


class Inst(ctypes.Structure):
    _fields_ = [
        ("cat_stall", ctypes.c_uint32),
        ("duration", ctypes.c_int32),
        ("time", ctypes.c_int64),
        ("pc", PcInfo),
    ]

    @property
    def stall(self):
        return (self.cat_stall >> 8) & 0xFFFFFF


class WaveState(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("duration", ctypes.c_int32),
    ]


class Wave(ctypes.Structure):
    _fields_ = [
        ("cu", ctypes.c_uint8),
        ("simd", ctypes.c_uint8),
        ("wave_id", ctypes.c_uint8),
        ("contexts", ctypes.c_uint8),
        ("dispatcher", ctypes.c_uint8),
        ("workgroup_id", ctypes.c_uint8),
        ("reserved", ctypes.c_uint16),
        ("size", ctypes.c_uint64),
        ("begin_time", ctypes.c_int64),
        ("end_time", ctypes.c_int64),
        ("timeline_size", ctypes.c_uint64),
        ("instructions_size", ctypes.c_uint64),
        ("timeline_array", ctypes.POINTER(WaveState)),
        ("instructions_array", ctypes.POINTER(Inst)),
    ]


class Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


TRACE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p,
)

ISA_CB = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    PcInfo,
    ctypes.c_void_p,
)


@dataclass
class Instruction:
    address: int
    text: str
    mnemonic: str
    symbol: str
    size: int


@dataclass
class Event:
    time: int
    duration: int
    stall: int
    pc: tuple[int, int]
    instruction: Instruction | None


@dataclass
class DecodedWave:
    location: str
    cu: int
    simd: int
    wave_id: int
    workgroup_id: int
    begin_time: int
    end_time: int
    events: list[Event]


def instruction_from_text(address: int, text: str, symbol: str = "") -> Instruction:
    mnemonic = text.split()[0] if text.split() else ""
    return Instruction(
        address=address,
        text=text,
        mnemonic=mnemonic,
        symbol=symbol,
        size=4,
    )


def find_rocm_root() -> Path:
    env = os.environ.get("ROCM_ROOT")
    if env:
        return Path(env)
    local = Path("/home/benoit/workspace/TheRock-build/dist/rocm")
    if (local / "llvm/bin/llvm-objdump").exists():
        return local
    raise SystemExit("error: set ROCM_ROOT to a ROCm SDK containing llvm-objdump")


def find_decoder_lib() -> Path:
    env = os.environ.get("ROCPROF_TRACE_DECODER_LIB")
    if env:
        return Path(env)
    local = Path(
        "/home/benoit/workspace/TheRock-build/profiler/rocprofiler-sdk/dist/lib/"
        "librocprof-trace-decoder.so"
    )
    if local.exists():
        return local
    raise SystemExit("error: set ROCPROF_TRACE_DECODER_LIB to librocprof-trace-decoder.so")


def code_object_id(path: Path) -> int:
    match = re.search(r"code_object_id_(\d+)\.out$", path.name)
    if not match:
        raise SystemExit(f"error: cannot determine code object id from {path}")
    return int(match.group(1))


def parse_disassembly(path: Path, objdump: Path, arch: str) -> dict[int, Instruction]:
    result = subprocess.run(
        [str(objdump), "-d", "-C", f"--mcpu={arch}", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    instructions: list[Instruction] = []
    symbol = ""
    header_re = re.compile(r"^([0-9a-fA-F]+) <(.+)>:$")
    inst_re = re.compile(r"^\s*(.*?)\s+//\s+([0-9a-fA-F]+):")
    for line in result.stdout.splitlines():
        header = header_re.match(line)
        if header:
            symbol = header.group(2)
            continue
        inst = inst_re.match(line)
        if not inst:
            continue
        text = inst.group(1).strip()
        if not text:
            continue
        address = int(inst.group(2), 16)
        mnemonic = text.split()[0]
        instructions.append(
            Instruction(
                address=address,
                text=text,
                mnemonic=mnemonic,
                symbol=symbol,
                size=4,
            )
        )

    for i, instruction in enumerate(instructions):
        if i + 1 < len(instructions):
            next_instruction = instructions[i + 1]
            if next_instruction.address > instruction.address:
                instruction.size = next_instruction.address - instruction.address

    return {instruction.address: instruction for instruction in instructions}


def load_instruction_maps(att_dir: Path, arch: str) -> dict[int, dict[int, Instruction]]:
    rocm_root = find_rocm_root()
    objdump = rocm_root / "llvm/bin/llvm-objdump"
    if not objdump.exists():
        raise SystemExit(f"error: missing {objdump}")

    maps = {}
    for path in sorted(att_dir.rglob("*code_object_id_*.out")):
        maps[code_object_id(path)] = parse_disassembly(path, objdump, arch)
    if not maps:
        raise SystemExit(f"error: found no code_object_id_*.out files under {att_dir}")
    return maps


def load_decoder(path: Path):
    lib = ctypes.CDLL(str(path))
    lib.rocprof_trace_decoder_create_handle.argtypes = [ctypes.POINTER(Handle)]
    lib.rocprof_trace_decoder_create_handle.restype = ctypes.c_int
    lib.rocprof_trace_decoder_destroy_handle.argtypes = [Handle]
    lib.rocprof_trace_decoder_destroy_handle.restype = ctypes.c_int
    lib.rocprof_trace_decoder_set_isa_callback.argtypes = [Handle, ISA_CB, ctypes.c_void_p]
    lib.rocprof_trace_decoder_set_isa_callback.restype = ctypes.c_int
    lib.rocprof_trace_decoder_parse.argtypes = [
        Handle,
        ctypes.c_void_p,
        ctypes.c_uint64,
        TRACE_CB,
        ctypes.c_void_p,
    ]
    lib.rocprof_trace_decoder_parse.restype = ctypes.c_int
    lib.rocprof_trace_decoder_get_info_string.argtypes = [ctypes.c_int]
    lib.rocprof_trace_decoder_get_info_string.restype = ctypes.c_char_p
    return lib


def decode_att(att_files: list[Path], instruction_maps: dict[int, dict[int, Instruction]]) -> list[DecodedWave]:
    lib = load_decoder(find_decoder_lib())
    handle = Handle()
    status = lib.rocprof_trace_decoder_create_handle(ctypes.byref(handle))
    if status != STATUS_SUCCESS:
        raise SystemExit(f"error: create decoder handle failed with status {status}")

    waves: list[DecodedWave] = []
    callback_refs = []

    def lookup(pc: PcInfo) -> Instruction | None:
        per_object = instruction_maps.get(pc.code_object_id)
        if per_object is None:
            return None
        return per_object.get(pc.address)

    def isa_callback(instr_buf, memory_size, size, pc, _userdata):
        instruction = lookup(pc)
        if instruction is None:
            return STATUS_ERROR_INVALID_ARGUMENT
        encoded = instruction.text.encode()
        available = size[0]
        size[0] = len(encoded)
        memory_size[0] = instruction.size
        if len(encoded) > available:
            return STATUS_ERROR_OUT_OF_RESOURCES
        ctypes.memmove(instr_buf, encoded, len(encoded))
        return STATUS_SUCCESS

    isa_cb = ISA_CB(isa_callback)
    callback_refs.append(isa_cb)
    status = lib.rocprof_trace_decoder_set_isa_callback(handle, isa_cb, None)
    if status != STATUS_SUCCESS:
        raise SystemExit(f"error: set ISA callback failed with status {status}")

    def trace_callback(record_type, trace_events, trace_size, _userdata):
        if record_type == RECORD_INFO:
            infos = ctypes.cast(trace_events, ctypes.POINTER(ctypes.c_int))
            for i in range(trace_size):
                msg = lib.rocprof_trace_decoder_get_info_string(infos[i])
                if msg:
                    print(f"decoder warning: {msg.decode()}", file=sys.stderr)
            return STATUS_SUCCESS

        if record_type != RECORD_WAVE:
            return STATUS_SUCCESS

        wave_records = ctypes.cast(trace_events, ctypes.POINTER(Wave))
        for wave_index in range(trace_size):
            wave = wave_records[wave_index]
            events = []
            insts = ctypes.cast(wave.instructions_array, ctypes.POINTER(Inst))
            for inst_index in range(wave.instructions_size):
                inst = insts[inst_index]
                instruction = lookup(inst.pc)
                events.append(
                    Event(
                        time=int(inst.time),
                        duration=int(inst.duration),
                        stall=int(inst.stall),
                        pc=(int(inst.pc.code_object_id), int(inst.pc.address)),
                        instruction=instruction,
                    )
                )
            waves.append(
                DecodedWave(
                    location=f"cu{int(wave.cu)}_simd{int(wave.simd)}_slot{int(wave.wave_id)}",
                    cu=int(wave.cu),
                    simd=int(wave.simd),
                    wave_id=int(wave.wave_id),
                    workgroup_id=int(wave.workgroup_id),
                    begin_time=int(wave.begin_time),
                    end_time=int(wave.end_time),
                    events=events,
                )
            )
        return STATUS_SUCCESS

    trace_cb = TRACE_CB(trace_callback)
    callback_refs.append(trace_cb)
    try:
        for path in att_files:
            raw = path.read_bytes()
            data = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
            status = lib.rocprof_trace_decoder_parse(handle, data, len(raw), trace_cb, None)
            if status != STATUS_SUCCESS:
                raise SystemExit(f"error: parse failed for {path} with status {status}")
    finally:
        lib.rocprof_trace_decoder_destroy_handle(handle)
        callback_refs.clear()
    return waves


def decode_ui_json(att_dir: Path) -> list[DecodedWave]:
    """Read ROCprof's stitched per-wave UI JSON output.

    The raw ATT decoder is useful as a fallback, but the UI JSON is the most
    direct way to validate the instruction order that ROCprof itself reports.
    Each per-wave instruction tuple ends with an index into code.json.
    """
    waves: list[DecodedWave] = []
    for code_path in sorted(att_dir.rglob("ui_output_agent_*_dispatch_*/code.json")):
        dispatch_dir = code_path.parent
        code_json = json.loads(code_path.read_text())
        current_symbol = ""
        code: list[Instruction | None] = []
        code_rows = code_json.get("code")
        if not code_rows:
            continue
        for row in code_rows:
            if not row:
                code.append(None)
                continue
            text = str(row[0]).strip()
            address = int(row[5]) if len(row) > 5 and isinstance(row[5], int) else 0
            if text.startswith("; "):
                current_symbol = text[2:]
                code.append(None)
                continue
            code.append(instruction_from_text(address, text, current_symbol))

        for wave_path in sorted(dispatch_dir.glob("se*_sm*_sl*_wv*.json")):
            wave_json = json.loads(wave_path.read_text())
            wave_info = wave_json.get("wave", {})
            events: list[Event] = []
            for entry in wave_info.get("instructions", []):
                if len(entry) < 5:
                    continue
                code_index = int(entry[-1])
                instruction = code[code_index] if 0 <= code_index < len(code) else None
                events.append(
                    Event(
                        time=int(entry[0]),
                        duration=int(entry[3]),
                        stall=int(entry[2]),
                        pc=(0, instruction.address if instruction else 0),
                        instruction=instruction,
                    )
                )
            waves.append(
                DecodedWave(
                    location=wave_path.stem,
                    cu=int(wave_info.get("cu", 0)),
                    simd=int(wave_info.get("simd", 0)),
                    wave_id=int(wave_info.get("slot", wave_info.get("id", 0))),
                    workgroup_id=0,
                    begin_time=int(wave_info.get("begin", 0)),
                    end_time=int(wave_info.get("end", 0)),
                    events=events,
                )
            )
    return waves


def marker(event: Event) -> str | None:
    instruction = event.instruction
    if instruction is None:
        return None
    text = instruction.text
    mnemonic = instruction.mnemonic
    if mnemonic == "s_setprio":
        parts = text.split()
        if len(parts) >= 2:
            return f"P{parts[1].rstrip(',')}"
        return "P?"
    if mnemonic.startswith("v_wmma"):
        return "WMMA"
    if mnemonic.startswith("ds_write") or mnemonic.startswith("ds_store"):
        return "LDS-W"
    if mnemonic.startswith("ds_read") or mnemonic.startswith("ds_load"):
        return "LDS-R"
    if mnemonic.startswith("global_load") or mnemonic.startswith("flat_load"):
        return "LOAD"
    if mnemonic.startswith("global_store") or mnemonic.startswith("flat_store"):
        return "STORE"
    if mnemonic.startswith("scratch_"):
        return "SCRATCH"
    if mnemonic.startswith("s_barrier"):
        return "BARRIER"
    return None


def compact_markers(events: list[Event]) -> list[str]:
    result = []
    previous = None
    repeat = 0
    for event in events:
        current = marker(event)
        if current is None:
            continue
        if current == previous:
            repeat += 1
            continue
        if previous is not None:
            result.append(f"{previous}x{repeat}" if repeat > 1 else previous)
        previous = current
        repeat = 1
    if previous is not None:
        result.append(f"{previous}x{repeat}" if repeat > 1 else previous)
    return result


def count(events: list[Event], predicate) -> int:
    return sum(1 for event in events if event.instruction is not None and predicate(event.instruction))


def high_priority_lds_regions(events: list[Event]) -> int:
    regions = 0
    for i, event in enumerate(events):
        instruction = event.instruction
        if not instruction or not instruction.text.startswith("s_setprio 1"):
            continue
        next_priority_index = None
        for j in range(i + 1, len(events)):
            next_instruction = events[j].instruction
            if next_instruction and next_instruction.mnemonic == "s_setprio":
                next_priority_index = j
                break
        if next_priority_index is None:
            continue
        if any(marker(events[j]) in {"LDS-W", "LDS-R"} for j in range(i + 1, next_priority_index)):
            regions += 1
    return regions


def lds_priority_signature(events: list[Event]) -> str:
    signature = []
    current_priority = "?"
    in_lds_cluster = False
    for event in events:
        current = marker(event)
        if current == "P1":
            current_priority = "1"
            continue
        if current == "P0":
            current_priority = "0"
            continue
        if current in {"LDS-W", "LDS-R"}:
            if not in_lds_cluster:
                signature.append(current_priority)
                in_lds_cluster = True
            continue
        if current == "WMMA":
            in_lds_cluster = False
    return "".join(signature)


def validate_wave(wave: DecodedWave, expected_lds_priority_signature: str | None) -> list[str]:
    failures = []
    events = wave.events
    wmma_indexes = [
        i for i, event in enumerate(events) if event.instruction and event.instruction.mnemonic.startswith("v_wmma")
    ]
    if not wmma_indexes:
        return failures

    setprio1_count = count(events, lambda inst: inst.text.startswith("s_setprio 1"))
    setprio0_count = count(events, lambda inst: inst.text.startswith("s_setprio 0"))
    if setprio1_count == 0:
        failures.append("wave executes WMMA but never executes s_setprio 1")
    if setprio0_count == 0:
        failures.append("wave executes WMMA but never executes s_setprio 0")

    if not any(marker(event) == "LDS-W" for event in events):
        failures.append("wave executes WMMA but no LDS store is visible in the trace")
    if not any(marker(event) == "LDS-R" for event in events):
        failures.append("wave executes WMMA but no LDS load is visible in the trace")

    if high_priority_lds_regions(events) == 0:
        failures.append("wave executes WMMA but no s_setprio 1 region covers visible LDS traffic")

    signature = lds_priority_signature(events)
    if expected_lds_priority_signature and signature != expected_lds_priority_signature:
        failures.append(
            f"LDS priority signature is {signature}, expected {expected_lds_priority_signature}"
        )

    for wmma_index in wmma_indexes:
        last_priority = None
        last_priority_index = None
        for i in range(wmma_index - 1, -1, -1):
            instruction = events[i].instruction
            if instruction and instruction.mnemonic == "s_setprio":
                last_priority = instruction.text
                last_priority_index = i
                break
        if last_priority is None:
            failures.append("WMMA has no preceding s_setprio")
            continue
        if not last_priority.startswith("s_setprio 0"):
            failures.append(f"WMMA is preceded by {last_priority}, not s_setprio 0")
    return failures


def summarize(waves: list[DecodedWave], max_waves: int, expected_lds_priority_signature: str | None) -> int:
    wmma_waves = [wave for wave in waves if any(marker(event) == "WMMA" for event in wave.events)]
    print(f"decoded_waves={len(waves)} wmma_waves={len(wmma_waves)}")

    failures = []
    for wave in wmma_waves:
        wave_failures = validate_wave(wave, expected_lds_priority_signature)
        for failure in wave_failures:
            failures.append(
                f"{wave.location} cu={wave.cu} simd={wave.simd} wave={wave.wave_id} "
                f"workgroup={wave.workgroup_id}: {failure}"
            )

    for wave in wmma_waves[:max_waves]:
        setprio1 = count(wave.events, lambda inst: inst.text.startswith("s_setprio 1"))
        setprio0 = count(wave.events, lambda inst: inst.text.startswith("s_setprio 0"))
        wmma = count(wave.events, lambda inst: inst.mnemonic.startswith("v_wmma"))
        lds_write = count(
            wave.events,
            lambda inst: inst.mnemonic.startswith("ds_write") or inst.mnemonic.startswith("ds_store"),
        )
        lds_read = count(
            wave.events,
            lambda inst: inst.mnemonic.startswith("ds_read") or inst.mnemonic.startswith("ds_load"),
        )
        high_priority_lds = high_priority_lds_regions(wave.events)
        global_load = count(
            wave.events,
            lambda inst: inst.mnemonic.startswith("global_load") or inst.mnemonic.startswith("flat_load"),
        )
        scratch = count(wave.events, lambda inst: inst.mnemonic.startswith("scratch_"))
        signature = lds_priority_signature(wave.events)
        print(
            f"wave {wave.location} cu={wave.cu} simd={wave.simd} slot={wave.wave_id} "
            f"workgroup={wave.workgroup_id} inst={len(wave.events)} "
            f"setprio1={setprio1} setprio0={setprio0} wmma={wmma} "
            f"lds_write={lds_write} lds_read={lds_read} high_prio_lds={high_priority_lds} "
            f"lds_priority_signature={signature} load={global_load} scratch={scratch}"
        )
        print("  markers: " + " ".join(compact_markers(wave.events)[:80]))

    if not waves:
        failures.append("ATT decoded no waves")
    if not wmma_waves:
        failures.append("ATT decoded no waves executing WMMA")

    if failures:
        print("validation=FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("validation=PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("att_dir", type=Path, help="rocprofv3 ATT output directory")
    parser.add_argument("--arch", default=os.environ.get("OFFLOAD_ARCH", "gfx1201"))
    parser.add_argument("--max-waves", type=int, default=6)
    parser.add_argument(
        "--expected-lds-priority-signature",
        help="fail if WMMA waves do not use this per-LDS-cluster priority signature, such as 1010",
    )
    parser.add_argument(
        "--raw-decoder",
        action="store_true",
        help="use librocprof-trace-decoder directly instead of ROCprof UI JSON",
    )
    args = parser.parse_args()

    waves = [] if args.raw_decoder else decode_ui_json(args.att_dir)
    if waves:
        print("source=rocprof-ui-json")
    else:
        att_files = sorted(args.att_dir.rglob("*.att"))
        if not att_files:
            raise SystemExit(f"error: found no .att files under {args.att_dir}")
        print("source=raw-att-decoder")
        instruction_maps = load_instruction_maps(args.att_dir, args.arch)
        waves = decode_att(att_files, instruction_maps)
    return summarize(waves, args.max_waves, args.expected_lds_priority_signature)


if __name__ == "__main__":
    sys.exit(main())
