import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
for p in root.rglob("*"):
    if ".git" in p.parts or not p.is_file() or p.suffix not in {".sh", ".py", ".world", ".cc", ".md", ".txt"}:
        continue
    b = p.read_bytes()
    if b"\r\n" in b:
        p.write_bytes(b.replace(b"\r\n", b"\n"))
        print("fixed", p.relative_to(root))
