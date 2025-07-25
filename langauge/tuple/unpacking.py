tup = (10,20.0 ,"ek")
x,y,z = tup 
print(x,y,z)
# error if no of unpacking elements and variables are not equal
# if still want less variable then use *
 # unpacking using *
tup1 = (10,20,30,40,50,60)
x,*y = tup1 
print(x)
print(y)
*x,y = tup1 
print(x)
print(y)
x,*y,z = tup1 
print(x)
print(y)
print(z)

# joining tuple - 
# 1. using concatenatio -> +
# 2.


#    _                Tuple methods
#      1. index() - Returns the lowest index in tuple that obj appears , parameter is any obj of tuple
tup2 = (10,2,3,3,3,3,3,3,3,3,3)
print(tup2.index(3))
#      2. count() - Returns count of how many times obj occurs in tuple , parameter is any obj of tuple
print(tup2.count(3))