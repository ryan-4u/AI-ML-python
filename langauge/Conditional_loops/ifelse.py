# if else and elif
age = 2 
if age>20 :
    print("Senior")
elif age == 20:
    print("classmate")
else:
    print("junior")   

# nested loops
marks = 9
if marks>33:
    print("passes in exam") 
    if marks >80:
        print(" with Merit..")
    else:
        print(" without Merit..")   
else:
    print("failed in exam..") 

# case 
day = 0 
match day:
    case 1:
        print("Mon")
    case 2:
        print("Tues")
    case 3:
        print("Wednes")
    case 4:
        print("Thrus")
    case 5:
        print("Fri")
    case 6:
        print("satur")
    case 7:
        print("sunday") 
    case _:
        print("Incalid input")
