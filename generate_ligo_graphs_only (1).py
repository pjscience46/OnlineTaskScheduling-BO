r"""Generate Pegasus LIGO DAGs beside this Python file.

No task CSV generation. Uses the compiled Pegasus WorkflowGenerator
(`bharathi` Java AppGenerator) to generate real LIGO-inspired DAX instances.

Default output (no matter which directory PowerShell is opened in):
  <folder_containing_this_script>\LIGO\
    GRAPHS\n=100\Ligo_output_1.csv
    GRAPHS\n=100\Ligo_output_2.csv
    ... and EMPTY n=200 ... n=1000 folders
    Tasks\n=100\                   (empty; no task parameter files)
    ORIGINAL\n=100\Ligo_output_1.dax
    ORIGINAL\n=100\Ligo_output_2.dax

The .csv graphs deliberately contain DOT-style edge lines because your
existing DAGGEN dependency reader expects them.

The precompiled Pegasus generator can REMAIN in:
  C:\Thesis\Paper2\JPDC_Reviews\WorkflowGenerator\bharathi
(or specify another location using --generator-root).

TEST:
  python generate_ligo_graphs_only.py --n 100 --count 2
MORE:
  python generate_ligo_graphs_only.py --n 100 --count 98 --start 3
  python generate_ligo_graphs_only.py --n 200 --count 2
  python generate_ligo_graphs_only.py --scaffold-only

Only Python's standard library is needed. Java is needed for generating DAX.
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
DATASET_FOLDER = SCRIPT_FOLDER / "LIGO"

# This is where you already cloned/compiled the Pegasus generator.
# It is an INPUT program, not the dataset output folder. Keep it as-is,
# or override from the command line: --generator-root <other_path>
GENERATOR_FOLDER = Path(r"C:\Thesis\Paper2\JPDC_Reviews\WorkflowGenerator\bharathi")

# Keep the same graph-size subfolders as your DAGGEN dataset.
GRAPH_SIZES = list(range(100, 1001, 100))
DEFAULT_N = 100
DEFAULT_COUNT = 2        # Only two graphs for the first test.
DEFAULT_START = 1        # Files named Ligo_output_1.csv, Ligo_output_2.csv, ...


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
    """Run actual Pegasus LIGO generator (it prints DAX XML to stdout)."""
    cmd = ["java", "-cp", generator_classpath(generator_path),
           "simulation.generator.AppGenerator", "-a", "LIGO", "-n", str(n)]
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
    raise RuntimeError(f"Pegasus LIGO failed after {retries} attempts: {previous_error}")


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
    parser = argparse.ArgumentParser(description="Generate LIGO DAGs in DAGGEN GRAPHS folder structure")
    parser.add_argument("--root", type=Path, default=DATASET_FOLDER,
                        help="Output LIGO folder")
    parser.add_argument("--generator-root", type=Path, default=GENERATOR_FOLDER,
                        help="Path to WorkflowGenerator/bharathi")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="Tasks per workflow (even and >20 for legacy LIGO generator)")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Number of workflows to produce (default 2)")
    parser.add_argument("--start", type=int, default=DEFAULT_START,
                        help="First file number (default 1)")
    parser.add_argument("--retries", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate original DAXs and overwrite graphs")
    parser.add_argument("--convert-only", action="store_true",
                        help="Use existing ORIGINAL .dax files without Java")
    parser.add_argument("--scaffold-only", action="store_true",
                        help="Only create the complete folder structure")
    args = parser.parse_args()

    if args.n not in GRAPH_SIZES:
        parser.error(f"--n must be one of {GRAPH_SIZES} for this folder layout")
    if args.n <= 20 or args.n % 2:
        parser.error("Pegasus LIGO requires an even job count greater than 20")
    if args.count <= 0 or args.start < 1 or args.retries < 1:
        parser.error("--count, --start and --retries must be positive")
    if args.convert_only and args.overwrite:
        parser.error("Use either --convert-only or --overwrite, not both")

    prepare_folders(args.root)
    print(f"Created/verified GRAPHS, Tasks (empty), and ORIGINAL subfolders for n=100...1000 under {args.root}")
    if args.scaffold_only:
        print("Folder structure created. No graphs generated.")
        return

    failures = 0
    for index in range(args.start, args.start + args.count):
        original = args.root / "ORIGINAL" / f"n={args.n}" / f"Ligo_output_{index}.dax"
        graph = args.root / "GRAPHS" / f"n={args.n}" / f"Ligo_output_{index}.csv"
        try:
            if graph.exists() and not args.overwrite:
                print(f"[SKIPPED] {graph.name} already exists (use --overwrite to replace)")
                continue
            if original.exists() and not args.overwrite:
                xml_text = original.read_text(encoding="utf-8-sig")
                print(f"Using existing original: {original.name}")
            elif args.convert_only:
                raise FileNotFoundError(f"Missing original: {original}")
            else:
                xml_text = generate_dax(args.generator_root, args.n, args.retries)

            graph_text, edge_count = convert_dax(xml_text, args.n)
            if not original.exists() or args.overwrite:
                original.write_text(xml_text, encoding="utf-8")
            graph.write_text(graph_text, encoding="utf-8")
            print(f"[OK] {graph.name}: n={args.n} tasks, {edge_count} dependencies")
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {graph.name}: {exc}", file=sys.stderr)
    if failures:
        sys.exit(f"Finished with {failures} failure(s). See messages above.")
    print("Completed. No task parameter files were generated.")


if __name__ == "__main__":
    main()
