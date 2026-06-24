import os
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def finalize(self, version, build_data, artifact_path):
        os.makedirs("sbom", exist_ok=True)
        output_file = os.path.join("sbom", "sbom.cyclonedx.json")
        try:
            subprocess.run(
                [
                    "cyclonedx-py",
                    "environment",
                    "--output-format", "JSON",
                    "--output-file", output_file,
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
