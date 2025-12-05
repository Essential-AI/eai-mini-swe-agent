#!/usr/bin/env python3
"""
Simple trajectory inspector for browsing agent conversation trajectories.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/inspector/[/bold green]
[/not dim]
"""

import json
import os
from pathlib import Path

import typer
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from minisweagent.agents.interactive_textual import _messages_to_steps

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class TrajectoryInspector(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "Scroll down"),
        Binding("k,up", "scroll_up", "Scroll up"),
        Binding("L", "next_trajectory", "Next trajectory"),
        Binding("H", "previous_trajectory", "Previous trajectory"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, trajectory_files: list[Path]):
        css_path = os.environ.get(
            "MSWEA_INSPECTOR_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "mini.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()

        super().__init__()
        self.trajectory_files = trajectory_files
        self._i_trajectory = 0
        self._i_step = 0
        self.messages = []
        self.steps = []

        if trajectory_files:
            self._load_current_trajectory()

    # --- Basics ---

    @property
    def i_step(self) -> int:
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step and self.n_steps > 0:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_steps(self) -> int:
        """Number of steps in current trajectory."""
        return len(self.steps)

    @property
    def i_trajectory(self) -> int:
        """Current trajectory index."""
        return self._i_trajectory

    @i_trajectory.setter
    def i_trajectory(self, value: int) -> None:
        """Set current trajectory index, automatically clamping to valid bounds."""
        if value != self._i_trajectory and self.n_trajectories > 0:
            self._i_trajectory = max(0, min(value, self.n_trajectories - 1))
            self._load_current_trajectory()
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_trajectories(self) -> int:
        """Number of trajectory files."""
        return len(self.trajectory_files)

    def _load_current_trajectory(self) -> None:
        """Load the currently selected trajectory file."""
        if not self.trajectory_files:
            self.messages = []
            self.steps = []
            return

        trajectory_file = self.trajectory_files[self.i_trajectory]
        try:
            data = json.loads(trajectory_file.read_text())

            if isinstance(data, list):
                self.messages = data
            elif isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
            else:
                raise ValueError("Unrecognized trajectory format")

            self.steps = _messages_to_steps(self.messages)
            self._i_step = 0
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            self.messages = []
            self.steps = []
            self.notify(f"Error loading {trajectory_file.name}: {e}", severity="error")

    @property
    def current_trajectory_name(self) -> str:
        """Get the name of the current trajectory file."""
        if not self.trajectory_files:
            return "No trajectories"
        return self.trajectory_files[self.i_trajectory].name

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()

    def update_content(self) -> None:
        """Update the displayed content."""
        container = self.query_one("#content", Vertical)
        container.remove_children()

        if not self.steps:
            container.mount(Static("No trajectory loaded or empty trajectory"))
            self.title = "Trajectory Inspector - No Data"
            return

        for message in self.steps[self.i_step]:
            if isinstance(message["content"], list):
                content_str = "\n".join([item["text"] for item in message["content"]])
            else:
                content_str = str(message["content"])
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message["role"].replace("assistant", "mini-swe-agent")
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        self.title = (
            f"Trajectory {self.i_trajectory + 1}/{self.n_trajectories} - "
            f"{self.current_trajectory_name} - "
            f"Step {self.i_step + 1}/{self.n_steps}"
        )

    # --- Navigation actions ---

    def action_next_step(self) -> None:
        self.i_step += 1

    def action_previous_step(self) -> None:
        self.i_step -= 1

    def action_first_step(self) -> None:
        self.i_step = 0

    def action_last_step(self) -> None:
        self.i_step = self.n_steps - 1

    def action_next_trajectory(self) -> None:
        self.i_trajectory += 1

    def action_previous_trajectory(self) -> None:
        self.i_trajectory -= 1

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)

def explode_jsonl(path: str, temp_dir: str) -> None:
    # explode the jsonl file into different json files
    filename = os.path.basename(path)
    local_jsonl = os.path.join(temp_dir, filename)
    jsons = [json.loads(js) for js in open(local_jsonl).read().split("\n") if js.strip()]
    # write each json to a new file in the temp directory
    for i, js in enumerate(jsons):
        if "resolved" in js:
            # json is dump from traj_stats upload, and not a direct jsonl of trajectories
            id = js["id"]
            js = js["trajectory"]
        else:
            id = i
        with open(os.path.join(temp_dir, f"{id}.traj.json"), "w") as f:
            json.dump(js, f, indent=4)
    # remove the jsonl file
    os.remove(local_jsonl)

def download_gs_directory(path: str, cache_dir: str = "/var/tmp/mini-swe-agent-inspector") -> str:
    import logging
    import tempfile
    import subprocess
    # use subprocess to run gsutil -m cp -r gs://path/to/directory /tmp/directory
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # get the directory name from the path
    assert path.startswith('gs://')
    directory_name = path[len('gs://'): ]
    temp_dir = os.path.join(cache_dir, directory_name)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        logging.info(f"Cache {temp_dir} exists", flush=True)
        return temp_dir
    logging.info(f"Downloading to cache {temp_dir}", flush=True)
    subprocess.run(["gsutil", "-m", "cp", "-r", path, temp_dir], check=True)
    logging.info(f"Downloaded {path} to {temp_dir}", flush=True)

    # if the file is a jsonl file, explode it into different json files
    # they might be dump from traj_stats upload, and not a direct jsonl of trajectories
    # in which case extract the trajectory from the json and write it to a new file
    if path.endswith(".jsonl"):
        explode_jsonl(path, temp_dir)
        logging.info(f"Exploded {path} into different json files. Removed jsonl", flush=True)

    return temp_dir

@app.command(help=__doc__)
def main(
    path: str = typer.Argument(".", help="Directory to search for trajectory files or specific trajectory file"),
) -> None:
    # if path starts with gs:// download the directory contents to a local temporary directory
    if path.startswith("gs://"):
        path = download_gs_directory(path)

    path_obj = Path(path)

    if path_obj.is_file():
        trajectory_files = [path_obj]
    elif path_obj.is_dir():
        trajectory_files = sorted(path_obj.rglob("*.traj.json"))
        if not trajectory_files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
    else:
        raise typer.BadParameter(f"Error: Path '{path}' does not exist")

    inspector = TrajectoryInspector(trajectory_files)
    inspector.run()


if __name__ == "__main__":
    app()
