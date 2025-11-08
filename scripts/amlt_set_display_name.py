#!/usr/bin/env python
import typer
from typing import Optional, List
import os
import sys
import types

# Temporary shim for Python 3.13 removal of stdlib imghdr (azureml-core still imports it).
try:
    import imghdr  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    shim = types.ModuleType("imghdr")
    def _noop_what(file, h=None):  # minimal stub; azureml only needs presence
        return None
    shim.what = _noop_what  # type: ignore[attr-defined]
    sys.modules["imghdr"] = shim

from azureml.core import Workspace, Run, Experiment  # noqa: E402
from azure.identity import DefaultAzureCredential 

TAG_KEY = "amlt-exp-name"

def _get_workspace(subscription_id: str, resource_group: str, workspace_name: str) -> Workspace:
    # If you already have a config.json you could just do Workspace.from_config()
    return Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
    )

def _list_tagged_runs(
    ws: Workspace,
    tag_value: str,
    max_experiments: Optional[int] = None,
    first_only: bool = False,
    verbose: bool = False,
) -> List[Run]:
    """Enumerate runs matching tag value. If first_only is True, return immediately on first match."""
    matches: List[Run] = []
    exp_names = list(ws.experiments.keys())
    if max_experiments is not None:
        exp_names = exp_names[:max_experiments]
    for idx, name in enumerate(exp_names):
        exp = ws.experiments[name]
        if verbose:
            typer.echo(f"[scan] {idx+1}/{len(exp_names)} experiment={name}")
        try:
            for r in Run.list(exp, tags=[f"{TAG_KEY}={tag_value}"]):
                matches.append(r)
                if first_only:
                    return matches
        except Exception:
            for r in Run.list(exp):
                if (r.get_tags() or {}).get(TAG_KEY) == tag_value:
                    matches.append(r)
                    if first_only:
                        return matches
    return matches

def batch_set_display_name_by_tags(
    new_display_name: str = typer.Argument(..., help="Display name to apply to all matching runs."),
    amlt_job_names: List[str] = typer.Argument(..., help="One or more amlt-exp-name tag values."),
    description: Optional[str] = typer.Option(None, help="Optional description property."),
    subscription_id: str = typer.Option(..., envvar="AZ_SUBSCRIPTION_ID"),
    resource_group: str = typer.Option(..., envvar="AZ_RESOURCE_GROUP"),
    workspace_name: str = typer.Option(..., envvar="AZ_WORKSPACE_NAME"),
    max_experiments: Optional[int] = typer.Option(50, help="Limit experiment scan count (default: 50, use 0 for unlimited)."),
    latest_only: bool = typer.Option(True, help="For each tag, update only the latest matching run."),
    dry_run: bool = typer.Option(False, help="Show what would be updated without making changes."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show scan progress."),
):
    """Update display name for runs matching multiple amlt-exp-name tag values in batch."""
    ws = _get_workspace(subscription_id, resource_group, workspace_name)
    max_exp_limit = None if max_experiments == 0 else max_experiments
    
    total_updated = 0
    total_found = 0
    
    for job_name in amlt_job_names:
        if verbose:
            typer.echo(f"[processing] amlt-exp-name={job_name}")
        runs = _list_tagged_runs(ws, job_name, max_experiments=max_exp_limit, first_only=latest_only, verbose=False)
        if not runs:
            typer.echo(f"[skip] No runs found for {job_name}")
            continue
            
        total_found += len(runs)
        
        if latest_only and len(runs) > 1:
            # Sort and take only the latest
            def ts(r: Run):
                md = r.get_details() or {}
                props = md.get("runDefinition", {})
                for k in ("startTimeUtc", "createdUtc"):
                    v = md.get(k) or props.get(k)
                    if v:
                        return v
                return r.id
            runs.sort(key=ts, reverse=True)
            runs = [runs[0]]  # Keep only the latest
        
        if dry_run:
            for r in runs:
                typer.echo(f"[dry-run] Would update: job_name={job_name} run_id={r.id} current_display={r.display_name}")
            continue
            
        for r in runs:
            try:
                if r.display_name == new_display_name:
                    typer.echo(f"[skip] {job_name}/{r.id} already has display name '{new_display_name}'")
                    continue

                r.display_name = new_display_name
                if description:
                    r.add_properties({"description": description})
                r.flush()
                total_updated += 1
                typer.echo(f"[ok] {job_name} -> {r.id}")
            except Exception as e:  # noqa: BLE001
                typer.echo(f"[warn] Failed {job_name}/{r.id}: {e}")
    
    if dry_run:
        typer.echo(f"[dry-run] Would update {total_found} run(s) across {len(amlt_job_names)} job name(s).")
    else:
        typer.secho(f"Batch complete. Updated {total_updated}/{total_found} runs across {len(amlt_job_names)} job name(s).", fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(batch_set_display_name_by_tags)