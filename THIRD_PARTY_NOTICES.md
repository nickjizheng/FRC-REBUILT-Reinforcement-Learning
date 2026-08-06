# Third-party notices

The project MIT license covers only original project-authored source code and
documentation. Nothing in that license grants rights in the excluded material
listed below. Inclusion or reference does not imply affiliation, sponsorship,
or endorsement.

## Bundled parser dependency

`tools/_vendor/olefile/**` contains olefile 0.47, copyright Philippe Lagadec,
under its BSD license, together with code derived from the Python Imaging
Library under the PIL license. The complete notices are retained in that
directory, including `tools/_vendor/olefile/LICENSE.txt`.

## NVIDIA software

NVIDIA Isaac Sim and Isaac Lab are external development dependencies and are
not distributed under this project's MIT license. Isaac Lab is principally
BSD-3-Clause, while Isaac Sim, Omniverse Kit, models, textures, dependencies,
and other materials can carry additional terms. Users are responsible for
reviewing the applicable [Isaac Sim license files](https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html),
[additional software and materials terms](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/common/license-isaac-sim-additional.html),
and the [Isaac Lab license](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE).

## FRC-derived material

`assets/fresh_frc/**` contains interoperability data extracted from an FRC
Simulator installation. It is excluded from this project's MIT license. No
open redistribution grant for that material is asserted here. Verify the
original owner's terms and obtain any required permission before copying or
redistributing it.

## Robot CAD and derived assets

The following paths contain supplied CAD, vendor components, exports, or
derived geometry and are excluded from this project's MIT license:

- `assets/robot_reference_source/**`
- `assets/robot_reference/**`
- `assets/robot_runtime/**`

Some CAD files identify third-party manufacturers and component vendors. The
eDrawings export also embeds software and notices from Dassault Systèmes and
other libraries. Retain all embedded notices and verify the rights for each
source asset before redistribution.

## Documentation media and trademarks

`docs/images/rebuilt-isaac-sim.jpg` and `docs/media/onboard-policy-rollout.gif`
are documentation captures. The project MIT license does not grant rights in
the depicted interfaces, third-party assets, names, logos, or trademarks.

FIRST®, FIRST® Robotics Competition, and FRC® are registered trademarks of
FIRST®. This independent project is not overseen, involved with, or endorsed by
FIRST®. See the [FIRST trademark guidelines](https://www.firstinspires.org/sites/default/files/uploads/resource_library/brand/2024-season/first-trademark-guidelines-web.pdf).

## Questions about redistribution

When the ownership or license of a non-code asset is unclear, treat it as not
covered by the MIT license and seek permission from the relevant owner before
redistributing it. A code-only distribution that requires users to import
their own legitimately obtained assets is the safest packaging model.
