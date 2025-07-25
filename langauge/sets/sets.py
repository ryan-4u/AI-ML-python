                                        #CREATING SETS
# 1. using { }
set0 = {1,2,"aar",True}
print(set0)
# 2. using set()
set2 = set((1,2,3,4,5))
print(set2)
# -> duplicate elements are removed 
# -> can contain differenet datatypes , even other sets

                                        # ACCCESSING SETS
# -> accessing sets is not possible using index - they are unordered
# -> using for loop , list comprehension , using subset

                                        # ADDING ELEMENTS
# 1. using add() method - give element to be added in parameter ~ can only add single element
set_X = {1,2,5.0,"aaa"}
print(set_X)
set_X.add(False)
print(set_X)
# 2. using update() - give elements or swquense as parameer ~ to add multiple elements or sequece of elements
set_Y = { 10 , 89.0 , 1 , "CCC"} 
set_X.update(set_Y)
print(set_X)
# -> a different example
str = set("Hello")
str.update("World")
print(str)
# 3. using union - can use union() method [ another set as parameter] OR use '|' bwtween two sets 
setA = { 1,2,3,4,5}
setB = { 5,6,7,8,9}
setC = { 7,8,9,10}
joinedSet1 = setA.union(setB)
joinedSet2 = setB | setC
print(joinedSet1)
print(joinedSet2)
# 4. using set comprehension

                                        #REMOVE ELEMENTS
# 1. remove() - element as parameter
setA.remove(4)
print(setA)
# 2.  discard() - same as remove() , diff - dont produce error even if element is not present
setA.discard(3)
print(setA)
# 3. pop() - no parameter , remove an arbitary element from set , give error if empty set
setA.pop()
print(setA)
# 4. clear() - to remove all element and make that a empty set
setA.clear()
print(setA)

# 5. difference_update() OR -= -> to remove common element of two set -> the parameters and 
#                                 the set1 common element are removed and remaining set1's element 
#                                 are returned and change is done in real set1
set1 = {1,2,3,4,5,6,7}
set2 = { 1,3}
# set1 -= set2
# print(set1)

# 6. symmetric_difference or ^ -> return all element of set1 and set2 except for their commons 
#                                 change is not done in real set1 , its a new resultant set
set3 = set1.symmetric_difference(set2)
print(set3)

# 7. interection_update -> returns only common element of both set , modify set1
# set1.intersection_update(set2)
# print(set1)
# 8. intersection() -> same as 7 , just not modify , its in a resultant set
set4 = set1.intersection(set2)
print(set4)

# 9. symmetric_difference_update -> same as 6 just modify the actual set
set1.symmetric_difference_update(set2)
print(set1)

                                    #MORE METHODS
#  .isdisjoint() -> Returns True if two sets have a null intersection.
set7 = { 5,6,7,8}
set8 = { 1,2,3,4,5,6}
print(set7.isdisjoint(set8))
# set1.issubset(set2) -> return true if set1 is a subset of set2
set9 = {1,2,3}
x = set9.issubset(set8)
print(x)
# set1.issuperset(set2) -> return true if set1 is superset of set2
y = set8.issuperset(set9)
print(y)
# set1.difference(set2) -> return set1 with only distinct elements then set 2
z = set7.difference(set8)
print(z)