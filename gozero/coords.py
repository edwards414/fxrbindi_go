"""GTP <-> pgx coordinate conversion.

pgx Go: action = row * size + col, row 0 at the TOP of the board (verified
empirically: action 0 renders at the top-left corner).  Pass = size * size.
GTP: column letters A.. skipping I, left to right; row numbers 1.. bottom up.
"""
GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # no 'I'


def gtp_to_action(vertex: str, size: int) -> int:
    v = vertex.strip().upper()
    if v == "PASS":
        return size * size
    col = GTP_COLS.index(v[0])
    row = int(v[1:])
    if not (0 <= col < size and 1 <= row <= size):
        raise ValueError(f"vertex {vertex} out of range for size {size}")
    return (size - row) * size + col


def action_to_gtp(action: int, size: int) -> str:
    if action == size * size:
        return "pass"
    row, col = divmod(int(action), size)
    return f"{GTP_COLS[col]}{size - row}"
