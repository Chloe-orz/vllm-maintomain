# LWD launch configuration examples

These are configuration-only migration examples. They do not implement
distributed edge/cloud inference. Use the original centralized launch for
inference regression, not these templates as proof of edge/cloud support.

The `2e1c` directory models two independent edge instances on 76.76.26.17 and
one eight-card cloud instance on 76.76.26.232. `topology.yaml` has DP=1;
`topology_2dp.yaml` demonstrates DP=2. Each generator invocation writes one file.

Generate the DP=1 topology:

```bash
python tools/gen_lwd_topology.py \
    --scene cloud_share \
    --edge-machines '0@76.76.26.17,1@76.76.26.17' \
    --cloud-machines 76.76.26.232 \
    --dp 1 --port-base 15783 \
    -o etc/lwd/2e1c/topology.yaml
```

The legacy `enable_weight_nz_layout` flag is replaced by this baseline's
`weight_nz_mode: 1`. The scripts keep launch parameters inline in `--additional-config`. Copy the
same topology contents to `/home/a00954194/2e1c/topology.yaml` on each host, or
change that path directly in the scripts. E0/E1 use physical NPUs 6/7 and API
ports 8094/8095 in separate shell sessions. These physical indices never appear
in generator inputs or topology ranks.

The cloud launch file is a template: its model path, interface and physical
card list were not present in the supplied edge-only example and must be set
locally. Its eager/batch settings are examples, not performance recommendations.
The DP=1 scripts do not silently become DP=2 launchers by changing the YAML;
native launch settings must also match the chosen deployment when execution
support is migrated. This change does not implement that runtime behavior.

Original scene examples use these generator inputs, with a single `--dp 1`
or `--dp 2` and a distinct output filename:

| Scene | `--edge-machines` | `--cloud-machines` |
| --- | --- | --- |
| `single_instance` | `10.0.0.1` | `10.1.0.1` |
| `edge_share` | `10.0.0.1` | `10.1.0.1,10.1.0.2` |
| `cloud_share` | `10.0.0.1,10.0.0.2` | `10.1.0.1` |
| `lwd_cluster` | `10.0.0.1,10.0.0.2` | `10.1.0.1,10.1.0.2` |

See [configuration design](../../docs/design/lwd_configuration.md) for the
scope, source-commit mapping and validation commands.
