logic needed::
i for each block (9 cubblockses in total)
j for each cube (9 cubes in each block)

using backstarcking algorithm:
    (mainly recursion)
if num==0
then num++ and call the function again
else check if num is valid in the current position
    if valid
        place the number
        call the function again
        if solved return true
        else remove the number and try next num
    else
        num++
        call the function again