#!/usr/bin/env python3
"""Package library sources and bundled dependencies for a host build."""

import argparse
from datetime import datetime
import io
import os
from pathlib import Path
import subprocess
import tarfile


REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = Path("stable-diffusion.cpp")
OMITTED_SUBMODULES = {
    "examples/server/frontend",
    "thirdparty/libwebm",
    "thirdparty/libwebp",
    "thirdparty/llama.cpp",
}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def source_files(repo, paths, omitted=()):
    if Path(os.fsdecode(git(repo, "rev-parse", "--show-toplevel")).strip()).resolve() != repo:
        raise RuntimeError(f"Submodule not initialized: {repo}")
    entries = git(repo, "ls-files", "--stage", "-z", "--", *paths)
    files = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, _, stage = metadata.split()
        relative = os.fsdecode(name)
        if stage != b"0":
            raise RuntimeError(f"Unresolved merge: {relative}")
        if relative in omitted:
            continue
        path = repo / relative
        if mode == b"160000":
            files.extend(source_files(path.resolve(), ["."]))
        elif not path.is_file() and not path.is_symlink():
            raise RuntimeError(f"Tracked source is missing: {path}")
        elif path.is_symlink() and (
            os.path.isabs(os.readlink(path)) or not path.resolve().is_relative_to(REPO_ROOT)
        ):
            raise RuntimeError(f"Source symlink is absolute or points outside the repository: {path}")
        else:
            files.append(path)
    if not files:
        raise RuntimeError(f"No source files found in {repo}: {paths}")
    return files


def archive_metadata(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def package_sources(output_dir, with_runners=False):
    dependency = REPO_ROOT / "thirdparty/llama.cpp"
    paths = ["CMakeLists.txt", "LICENSE", "cmake", "include", "src", "thirdparty"]
    if with_runners:
        paths.append("examples")
    files = source_files(
        REPO_ROOT,
        paths,
        OMITTED_SUBMODULES,
    )
    files.extend(source_files(dependency, ["ggml", "LICENSE", "AUTHORS"]))
    version = os.fsdecode(git(REPO_ROOT, "describe", "--always", "--dirty")).strip()
    commit = os.fsdecode(git(REPO_ROOT, "rev-parse", "HEAD")).strip()
    dependency_commit = os.fsdecode(git(dependency, "rev-parse", "HEAD")).strip()
    contents = (
        "Server and CLI sources are included. Web UI, models, build outputs and Git metadata are excluded."
        if with_runners else
        "Examples, Web UI, models, build outputs and Git metadata are excluded."
    )
    build_flags = (
        "-DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF"
        if with_runners else "-DSD_BUILD_EXAMPLES=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF"
    )
    instructions = f"""stable-diffusion.cpp source archive
Version: {version}
Commit: {commit}
llama.cpp-sm_61 commit: {dependency_commit}

Contains tracked working-tree sources, including local modifications.
{contents}
The bundled GGML is thirdparty/llama.cpp/ggml; no submodule download is needed.

CPU build (requires CMake and a C/C++ compiler):
  cmake -S . -B build {build_flags}
  cmake --build build --parallel

For CUDA/sm_61, add -DSD_CUDA=ON to the configure command (requires CUDA Toolkit).
For a shared library, add -DSD_BUILD_SHARED_LIBS=ON.
""".encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"stable-diffusion.cpp-{datetime.now():%Y%m%d%H%M}.tar.gz"
    with archive_path.open("xb") as output:
        try:
            with tarfile.open(fileobj=output, mode="w:gz", compresslevel=6) as archive:
                for path in sorted(set(files)):
                    archive.add(
                        path,
                        arcname=str(ARCHIVE_ROOT / path.relative_to(REPO_ROOT)),
                        recursive=False,
                        filter=archive_metadata,
                    )
                info = tarfile.TarInfo(str(ARCHIVE_ROOT / "BUILD.txt"))
                info.size = len(instructions)
                info.mtime = int(datetime.now().timestamp())
                archive.addfile(info, io.BytesIO(instructions))
        except BaseException:
            archive_path.unlink()
            raise
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-runners", action="store_true",
        help="include sd-server and sd-cli sources, without the server Web UI",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT,
        help="archive output directory (default: repository root)",
    )
    args = parser.parse_args()
    try:
        print(package_sources(args.output_dir.resolve(), with_runners=args.with_runners))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
