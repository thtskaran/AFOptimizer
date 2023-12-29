for it to work corrrectlyyy under replit.nix add dependency

{ pkgs }: {
  deps = [
    pkgs.postgresql
    pkgs.libGL
  ];
}
