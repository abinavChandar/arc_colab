# full module below
def transform(grid: list[list[int]]) -> list[list[int]]:
    H, W = len(grid), len(grid[0])
    output_grid = [[0] * 9 for _ in range(9)]
    
    for i in range(H):
        for j in range(W):
            for k in range(3):
                for l in range(3):
                    output_grid[i*3 + k][j*3 + l] = grid[i][j]
                    
    return output_grid