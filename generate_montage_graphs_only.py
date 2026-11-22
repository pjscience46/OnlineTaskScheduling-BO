r"""Generate Pegasus Montage graph files for ALL graph sizes, beside this Python file.

Default (no flags): 100 Montage graphs per size, n=100,200,...,1000
=> 1,000 workflows across 10 size folders.

Files are saved alongside this script:
    MONTAGE/ORIGINAL/n=100/Montage_output_1.dax ... Montage_output_100.dax
    MONTAGE/GRAPHS/n=100/Montage_output_1.csv ... Montage_output_100.csv
    ... repeated for n=200 through n=1000
    MONTAGE/Tasks/n=...   (empty placeholders; NO task files are generated)

The .csv files contain DAGGEN-compatible DOT-style lines because your
existing dependency reader expects those, not standard comma-separated edges.

Source: Pegasus WorkflowGenerator (legacy bharathi Java Montage generator).
The generator may stay installed at:
  C:\Thesis\Paper2\JPDC_Reviews\WorkflowGenerator\bharathi

Commands:
  python generate_montage_graphs_only.py                  # full 1000 workflows
  python generate_montage_graphs_only.py --test           # 2 workflows at n=100 only
  python generate_montage_graphs_only.py --count 2        # 2 per size across all sizes
  python generate_montage_graphs_only.py --n 300 --count 100
  python generate_montage_graphs_only.py --convert-only   # convert saved DAX files

Already-generated graph files are skipped by default so interrupted generation
can be resumed safely by running the same command again.
"""

from __future__ import annotations

import argparse
import heapq
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Default output is based on the location of THIS script, not the terminal
# working directory. Moving this .py file moves the default output location.
SCRIPT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = SCRIPT_FOLDER / "MONTAGE"

# This is where you already cloned/compiled the Pegasus generator.
# It is an INPUT program, not the dataset output folder. Keep it as-is,
# or override from the command line: --generator-root <other_path>
GENERATOR_FOLDER = Path(r"C:\Thesis\Paper2\JPDC_Reviews\WorkflowGenerator\bharathi")

# Keep the same graph-size subfolders as your DAGGEN dataset.
GRAPH_SIZES = list(range(100, 1001, 100))
DEFAULT_COUNT = 100      # Full dataset: 100 files in EVERY n folder.
DEFAULT_START = 1        # Files named Montage_output_1.csv, Montage_output_2.csv, ...


def local_tag(element: ET.Element) -> str:
    """Handle Pegasus DAX XML namespaces."""
    return element.tag.rsplit("}", 1)[-1]


def prepare_folders(root: Path) -> None:
    """Create ALL n=100,200,...,1000 folders even for a two-graph test."""
    for n in GRAPH_SIZES:
        (root / "GRAPHS" / f"n={n}").mkdir(parents=True, exist_ok=True)
        (root / "Tasks" / f"n={n}").mkdir(parents=True, exist_ok=True)  # Empty: no task creation.
        (root / "ORIGINAL" / f"n={n}").mkdir(parents=True, exist_ok=True)


def parse_dax(xml_text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Read Pegasus jobs and exact parent -> child dependencies.

    Job ordering is made topological, so all edges go from a smaller to a
    larger task index. This ordering is stable for the same original DAX.
    """
    root = ET.fromstring(xml_text)
    if local_tag(root) != "adag":
        raise ValueError(f"Expected Pegasus <adag> root, got <{local_tag(root)}>")

    jobs = [e for e in root.iter() if local_tag(e) == "job"]
    if not jobs:
        raise ValueError("Pegasus DAX has no <job> elements")

    ids = [j.attrib.get("id") for j in jobs]
    if any(not job_id for job_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Pegasus DAX has missing or duplicate job IDs")

    job_order = {job_id: i for i, job_id in enumerate(ids)}
    outgoing: dict[str, set[str]] = {job_id: set() for job_id in ids}
    indegree = {job_id: 0 for job_id in ids}

    for child in root.iter():
        if local_tag(child) != "child":
            continue
        child_id = child.attrib.get("ref")
        if child_id not in job_order:
            raise ValueError(f"Unknown child job ID: {child_id}")
        for parent in child:
            if local_tag(parent) != "parent":
                continue
            parent_id = parent.attrib.get("ref")
            if parent_id not in job_order:
                raise ValueError(f"Unknown parent job ID: {parent_id}")
            if parent_id == child_id:
                raise ValueError("Self dependency is not a DAG")
            if child_id not in outgoing[parent_id]:
                outgoing[parent_id].add(child_id)
                indegree[child_id] += 1

    ready = [(job_order[j], j) for j in ids if indegree[j] == 0]
    heapq.heapify(ready)
    sorted_jobs: list[str] = []
    while ready:
        _, j = heapq.heappop(ready)
        sorted_jobs.append(j)
        for child_id in sorted(outgoing[j], key=lambda k: job_order[k]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, (job_order[child_id], child_id))

    if len(sorted_jobs) != len(ids):
        raise ValueError("The input workflow contains a cycle")

    # DOT graph edges are ONE-BASED, as with the DAGGEN files.
    numeric_id = {job_id: i + 1 for i, job_id in enumerate(sorted_jobs)}
    edges = sorted((numeric_id[p], numeric_id[c])
                   for p, children in outgoing.items() for c in children)
    return sorted_jobs, edges


def generator_classpath(path: Path) -> str:
    classes = path / "classes"
    libraries = sorted((path / "lib").glob("*.jar"))
    if not classes.is_dir() or not libraries:
        raise FileNotFoundError(
            "Pegasus Java generator not found or not compiled.\n"
            f"Expected: {path}\n"
            "Run: git clone https://github.com/pegasus-isi/WorkflowGenerator.git\n"
            "Then, inside WorkflowGenerator\\bharathi, run: ant compile"
        )
    return os.pathsep.join(str(p) for p in [classes, *libraries])


def generate_dax(generator_path: Path, n: int, retries: int) -> str:
    """Run actual Pegasus Montage generator (it prints DAX XML to stdout)."""
    cmd = ["java", "-cp", generator_classpath(generator_path),
           "simulation.generator.AppGenerator", "-a", "MONTAGE", "-n", str(n)]
    previous_error = "not attempted"
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", check=True)
            xml_text = result.stdout.strip()
            job_ids, _ = parse_dax(xml_text)
            if len(job_ids) != n:
                raise ValueError(f"Generator produced {len(job_ids)} jobs, expected {n}")
            return xml_text + "\n"
        except (OSError, subprocess.CalledProcessError, ValueError, ET.ParseError) as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                previous_error = (exc.stderr or exc.stdout or str(exc))[-1500:]
            else:
                previous_error = str(exc)
    raise RuntimeError(f"Pegasus Montage failed after {retries} attempts: {previous_error}")


def convert_dax(xml_text: str, expected_jobs: int) -> tuple[str, int]:
    job_ids, edges = parse_dax(xml_text)
    if len(job_ids) != expected_jobs:
        raise ValueError(f"Expected {expected_jobs} jobs; DAX contains {len(job_ids)}. "
                         "Graph not saved (jobs will not be added or removed).")

    # IMPORTANT: Existing DAGGEN extract_dependencies_from_csv() slices off
    # exactly 2 leading spaces, then parses 'parent -> child ' until a space.
    # No header, NO semicolon after the target, and a trailing space on edges.
    lines = ["digraph G {"]
    lines.extend(f"  {a} -> {b} " for a, b in edges)
    lines.append("}")
    return "\n".join(lines) + "\n", len(edges)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Montage graphs for n=100..1000, 100 instances each by default"
    )
    parser.add_argument("--root", type=Path, default=DATASET_FOLDER,
                        help="Output root MONTAGE (default: beside this Python script)")
    parser.add_argument("--generator-root", type=Path, default=GENERATOR_FOLDER,
                        help="Compiled WorkflowGenerator/bharathi location")
    parser.add_argument("--n", type=int, default=None,
                        help="Generate a single n size; omit to generate ALL 100..1000")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Number of graphs PER size (default: 100)")
    parser.add_argument("--start", type=int, default=DEFAULT_START,
                        help="First output file index (default: 1)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 2 graphs only at n=100")
    parser.add_argument("--retries", type=int, default=15,
                        help="Retries per graph if the legacy generator fails")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate original DAXs and overwrite CSV graphs")
    parser.add_argument("--convert-only", action="store_true",
                        help="Convert existing ORIGINAL DAXs without invoking Java")
    parser.add_argument("--scaffold-only", action="store_true",
                        help="Create the folders only; no DAX or graphs")
    args = parser.parse_args()

    if args.n is not None and args.n not in GRAPH_SIZES:
        parser.error(f"--n must be one of {GRAPH_SIZES}")
    if args.count <= 0 or args.start < 1 or args.retries < 1:
        parser.error("--count, --start and --retries must be positive")
    if args.convert_only and args.overwrite:
        parser.error("Do not combine --convert-only with --overwrite")

    # All size folders are created in every execution, even a small test.
    prepare_folders(args.root)
    print(f"Dataset location: {args.root}", flush=True)
    print("Created/verified GRAPHS, Tasks (empty), and ORIGINAL for n=100...1000", flush=True)
    if args.scaffold_only:
        print("Scaffold-only requested: no workflows generated")
        return

    if args.test:
        sizes = [100]
        count = 2
        start = 1
    else:
        sizes = [args.n] if args.n is not None else GRAPH_SIZES
        count = args.count
        start = args.start

    print(f"Graph sizes: {sizes}; {count} instances per size; total requested: {len(sizes)*count}", flush=True)
    print("Already-completed graph files will be skipped. Tasks folders stay empty.\n", flush=True)

    # Avoid calling Java repeatedly if Pegasus is not installed or compiled.
    # Only check the Java generator when a DAX will actually need generating.
    failures = 0
    created = 0
    skipped = 0
    for n in sizes:
        print(f"\n===== n={n} ({count} requested) =====", flush=True)
        ok_size, skip_size, fail_size = 0, 0, 0
        for index in range(start, start + count):
            original = args.root / "ORIGINAL" / f"n={n}" / f"Montage_output_{index}.dax"
            graph = args.root / "GRAPHS" / f"n={n}" / f"Montage_output_{index}.csv"
            if graph.exists() and not args.overwrite:
                skip_size += 1
                continue
            try:
                if original.exists() and not args.overwrite:
                    xml_text = original.read_text(encoding="utf-8-sig")
                elif args.convert_only:
                    raise FileNotFoundError(f"No original workflow found: {original}")
                else:
                    xml_text = generate_dax(args.generator_root, n, args.retries)

                graph_text, edge_count = convert_dax(xml_text, n)
                # Write original before graph; rerun will convert a saved DAX if needed.
                if not original.exists() or args.overwrite:
                    original.write_text(xml_text, encoding="utf-8")
                graph.write_text(graph_text, encoding="utf-8")
                ok_size += 1
                # Progress for large batches; avoid excessive 1000-line output.
                if ok_size <= 2 or ok_size % 10 == 0 or index == start + count - 1:
                    print(f"  [OK] {graph.name}: {n} jobs, {edge_count} edges", flush=True)
            except Exception as exc:
                fail_size += 1
                print(f"  [ERROR] n={n}, graph {index}: {exc}", file=sys.stderr, flush=True)
                # Missing generator is a nonrecoverable setup problem.
                if isinstance(exc, (FileNotFoundError, OSError)) and not args.convert_only:
                    raise SystemExit("Stopped: fix the Pegasus generator path/build, then rerun.")
        created += ok_size
        skipped += skip_size
        failures += fail_size
        print(f"n={n} finished: created={ok_size}, skipped={skip_size}, errors={fail_size}", flush=True)

    print(f"\nFINISHED: created={created}, skipped={skipped}, errors={failures}")
    print(f"Graph files: {args.root / 'GRAPHS'}")
    print("No task-parameter files were generated.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
