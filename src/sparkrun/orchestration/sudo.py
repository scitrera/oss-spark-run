"""Sudo fallback orchestration for sparkrun setup commands."""

from __future__ import annotations

from sparkrun.orchestration import ssh as _ssh


def run_with_sudo_fallback(
    host_list: list[str],
    script: str,
    fallback_script: str,
    ssh_kwargs: dict,
    dry_run: bool = False,
    sudo_password: str | None = None,
    timeout: int = 300,
) -> tuple[dict[str, object], list[str]]:
    """Run script with sudo fallback. Returns (result_map, still_failed_hosts).

    Steps:
    1. Try non-interactive sudo on all hosts in parallel.
    2. Partition into successes/failures.
    3. For failures, fall back to password-based sudo using provided password.
    4. Return result map and list of hosts that still failed after fallback.

    The CLI handler is responsible for prompting for passwords and handling
    per-host retries on still_failed_hosts.

    Args:
        host_list: Hosts to run on.
        script: Initial script to run (non-interactive sudo).
        fallback_script: Script for password-based sudo fallback.
        ssh_kwargs: SSH connection parameters.
        dry_run: If True, skip actual execution.
        sudo_password: Optional sudo password for fallback.
        timeout: Timeout in seconds for each operation.

    Returns:
        Tuple of (result_map, still_failed_hosts) where result_map is
        {host: SSHResult} and still_failed_hosts is a list of hosts
        that failed even after password-based sudo.
    """
    # Step 1: Try non-interactive sudo on all hosts in parallel
    parallel_results = _ssh.run_remote_scripts_parallel(
        host_list, script, timeout=timeout, dry_run=dry_run, **ssh_kwargs,
    )

    # Partition results: successes vs failures needing password
    result_map: dict[str, object] = {}
    failed_hosts = []
    for r in parallel_results:
        if r.success:
            result_map[r.host] = r
        else:
            failed_hosts.append(r.host)

    # Step 2: For failed hosts, fall back to password-based sudo
    if failed_hosts and not dry_run and sudo_password is not None:
        for h in failed_hosts:
            r = _ssh.run_remote_sudo_script(
                h, fallback_script, sudo_password, timeout=timeout, dry_run=dry_run, **ssh_kwargs,
            )
            result_map[h] = r

    # Return results and hosts that still failed after fallback
    still_failed = [h for h in failed_hosts if h not in result_map or not result_map[h].success]
    return result_map, still_failed
