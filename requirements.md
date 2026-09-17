Overview: Improve NMSTT through the Conductor execution loop.

Delivery Context:
- Current stage: development
- Validated stages: none
- Rollout strategy: canary

Requirements Register:
- REQ-001: Inspect and record current repository, runtime, or job evidence before selecting an operation.
- REQ-002: Implement only the scoped change, job update, or progress-monitoring action supported by that evidence.
- REQ-003: Preserve secure, resilient behaviour and avoid destructive commands.
- REQ-004: Update or add tests covering the changed path, or provide the relevant live operational check.
- REQ-005: Run verification commands and report the outcome.
- REQ-006: Leave unrelated files untouched.
- REQ-007: Record rollback/recovery steps and the acceptance signal proving the gap is closed.
- REQ-008: Preserve staged progression and rollout governance metadata.
- REQ-009: Capture a fresh protected-target readiness baseline before any change.
- REQ-010: Use the selected canary or red-green rollout strategy and verify the post-rollout health window.
- REQ-011: Automatically revert the exact produced commit without rewriting history if health or verification degrades.
- REQ-012: Verify rollback readiness and recovery before finalising the delivery.
- REQ-013: When runtime rollout or restart work is needed, use the available Ansible automation context: {"ansible_root":"/srv/swarmhpc/ansible","config_path":"/srv/swarmhpc/ansible/ansible.cfg","host_targets":["rk1"],"hosts":["spirit"],"inventory_path":"/srv/swarmhpc/ansible/inventory/hosts.ini","playbooks":["continuum_tenant_nmchain_site.yml","continuum_tenant_nmstt_site.yml"],"repo_root":"/srv/swarmhpc","roles_path":"/srv/swarmhpc/ansible/roles","secrets_root":"/srv/swarmhpc/ansible/.secrets"}.

Work Item Summary:
nmstt is linked to live services but no obvious test capability was discovered in the repository inventory. Establish at least a minimal regression or smoke-test baseline before deeper autonomous changes.

Authoritative delivery constraints (mandatory; implement and verify these, do not merely describe them):
- No structured delivery constraints were supplied; follow the work-item summary exactly.

Plan JSON:
{"action":"establish_repository_test_baseline","finding_id":"7a9b4c8d-f8ff-4ece-9b95-3c6ed0f621b3","finding_key":"repository_test_baseline:nmstt","linked_services":["nmstt"],"repository":"nmstt"}

Planner guidance (advisory; it must not weaken or contradict the authoritative work-item requirements):
Overview: This work item establishes a minimal regression or smoke-test baseline for the nmstt service, which is linked to live services but currently lacks discovered test capability. The objective is to secure a verifiable test foundation before deeper autonomous changes.

Requirements Register:
- REQ-001: Inspect the nmstt repository at /srv/neuralmimicry/nmstt to confirm path accessibility, branch state, and file structure.
- REQ-002: Query the local K3s cluster status using kubectl get nodes to verify control-plane connectivity and worker node probes.
- REQ-003: Review service health logs and metrics on host 'spirit' to identify specific degradation causes before proceeding with remediation.
- REQ-004: Execute ansible-playbook /srv/swarmhpc/ansible/playbooks/continuum_tenant_nmstt_site.yml --check to validate Ansible syntax and inventory without applying changes.
- REQ-005: Run cargo fmt --check and cargo check to verify Rust codebase integrity and compiler health.
- REQ-006: Execute cargo test to establish the current test baseline and capture passing fixtures.
- REQ-007: Create a minimal smoke-test suite targeting core nmstt functionality, ensuring coverage of at least one critical path.
- REQ-008: Update .gitignore or CI configuration to include the new test artifacts, then commit and push for review.

Rollout notes: Use canary strategy. Monitor for degradation during initial live impact. Automatic rollback on health signal loss.
Verification: Successful execution of cargo test, positive ansible --check, and observed runtime stability on 'spirit'.


Protected rollout contract (mandatory): capture a fresh readiness baseline before any change; use the selected canary or red_green strategy; verify health throughout the post-rollout window; if health or verification degrades, automatically revert the exact produced commit without rewriting history, rerun tests and GitHub Actions, and verify recovery.