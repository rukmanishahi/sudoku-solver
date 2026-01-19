def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end=" ")
        print()

def empty_location(arr, l):
    for row in range(9): #to scan rows and if 0 then position ill b saved [l]
        for col in range(9):
            if(arr[row][col]==0): 
                l[0] = row #saves rows
                l[1] = col #saves columns
                return True
    return False #no empty then false


#if num is alrd in a row
def used_in_row(arr, row, num):
    for i in range(9):
        if(arr[row][i] == num):
            return True
    return False

#if num used in column
def used_in_col(arr, col, num):
    for i in range(9):
        if(arr[i][col] == num):
            return True
    return False

#if num is in 1 block (3x3)
def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if(arr[i + row][j + col] == num):
                return True
    return False

#checks rows, col and block for num
def check_location(arr, row, col, num):
    return (not used_in_row(arr, row, num) and
            not used_in_col(arr, col, num) and
            not used_in_box(arr, row - row % 3,
                            col - col % 3, num))
