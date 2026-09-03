{
  description = "vtuber-vocal-corpus — VTuber chatting-stream voice measurements";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            # Never source-build CUDA torch/onnxruntime (disk bomb, see
            # CLAUDE.md). GPU acceleration comes from pip wheels:
            # onnxruntime-gpu + CUDA-tagged torch on x86_64-linux,
            # torch MPS / onnxruntime CoreML on aarch64-darwin.
            cudaSupport = false;
          };
        };
        inherit (pkgs) lib stdenv;

        python = pkgs.python312.withPackages (
          ps: with ps; [
            numpy
            scipy
            matplotlib
            requests
            pytest
            pip
          ]
        );

        # The pip wheels (onnxruntime-gpu, CUDA torch) load nixpkgs' cudnn /
        # cudatoolkit and the host GL driver at runtime. Linux only — on
        # darwin the loader ignores LD_LIBRARY_PATH and the macOS torch /
        # onnxruntime wheels bundle their own dylibs (Metal/CoreML), so this
        # attr is omitted there entirely.
        linuxLdLibraryPath =
          lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.cudatoolkit
          ]
          + ":/run/opengl-driver/lib";
      in
      {
        formatter = pkgs.nixfmt-rfc-style;

        devShells.default = pkgs.mkShell (
          {
            packages = [
              python
              pkgs.yt-dlp
              pkgs.deno
              pkgs.ffmpeg-headless
              pkgs.rclone
              pkgs.basedpyright
              pkgs.ruff
            ];

            venvDir = ".venv";

            shellHook = ''
              venvDir="''${venvDir:-.venv}"
              if [ ! -d "$venvDir" ]; then
                ${python}/bin/python -m venv --system-site-packages "$venvDir"
              fi
              # shellcheck disable=SC1091
              source "$venvDir/bin/activate"
              unset PYTHONPATH
              if ! python -c "import audio_separator" 2>/dev/null; then
                pip install -r requirements.txt
              fi
              venvVersionWarn() {
                local venvVersion
                venvVersion="$("$venvDir/bin/python" -c 'import platform; print(platform.python_version())')"
                [[ "$venvVersion" == "${pkgs.python312.version}" ]] && return
                cat <<EOF
              Warning: Python version mismatch: [$venvVersion (venv)] != [${pkgs.python312.version}]
                       Delete '$venvDir' and reload to rebuild for version ${pkgs.python312.version}
              EOF
              }
              venvVersionWarn
            '';
          }
          // lib.optionalAttrs stdenv.hostPlatform.isLinux {
            LD_LIBRARY_PATH = linuxLdLibraryPath;
          }
        );
      }
    );
}
