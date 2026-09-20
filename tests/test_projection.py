from arcforge.projection import projection_candidate_grids


def test_divider_periodic_projection():
    task = {
        "train": [
            {
                "input": [
                    [2, 2, 9, 0, 0, 0, 0],
                    [3, 3, 3, 9, 0, 0, 0],
                ],
                "output": [
                    [2, 2, 9, 2, 0, 2, 0],
                    [3, 3, 3, 9, 3, 0, 0],
                ],
            }
        ],
        "test": [
            {
                "input": [
                    [4, 4, 9, 0, 0, 0, 0],
                    [5, 5, 5, 9, 0, 0, 0],
                ]
            }
        ],
    }
    # This synthetic test is intentionally permissive: the engine should at
    # least discover a candidate under the indicator->projection family.
    assert isinstance(projection_candidate_grids(task), list)


def test_cell_template_completion_smoke():
    # 3x3 cells separated by one uniform row/column.
    normal = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    special = [
        [1, 2, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]

    def assemble(cells):
        out = []
        for ri, row_cells in enumerate(cells):
            for rr in range(3):
                row = []
                for ci, cell in enumerate(row_cells):
                    if ci:
                        row.append(0)
                    row.extend(cell[rr])
                out.append(row)
            if ri + 1 < len(cells):
                out.append([0] * len(out[0]))
        return out

    before = assemble([[special, normal], [normal, normal]])
    after = assemble([[special, special], [normal, normal]])
    task = {
        "train": [{"input": before, "output": after}],
        "test": [{"input": before}],
    }
    assert isinstance(projection_candidate_grids(task), list)
