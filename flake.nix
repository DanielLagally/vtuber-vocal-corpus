{
  description = "vanalysis — hololive chatting-stream voice measurements";

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
          config.allowUnfree = true;
        };

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
      in
      {
        formatter = pkgs.nixfmt-rfc-style;

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.yt-dlp
            pkgs.deno
            pkgs.ffmpeg-headless
            pkgs.basedpyright
            pkgs.ruff
          ];

          venvDir = ".venv";

          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cudatoolkit
            ]
            + ":/run/opengl-driver/lib";

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
        };
      }
    );
}
