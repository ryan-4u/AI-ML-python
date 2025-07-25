#                          Methods on list

l = [1,2,3,4,5,6,7]
# to access list items - use index , negative index and slicing method
print(l[2:])

#                          adding elements
# 1. append - to add element ( at end of list)
list =['a','b','c','d','e']
list.append('x')
print(list)
# 2 . insert - to add element at desired position 
list.insert(2,'y')
print(list)
# 3. extend - to join two lists
list_x = ['m','n','o']
list.extend(list_x)
print(list)

#                           removing elements
# 4. remove - to remove elements by giving parameter
list.remove(list[2])
print(list)
# 5. pop - to remove last element for no parameter , to remove parametered( only list index ) element
list.pop()
print(list)
list.pop(5)
print(list) 
# 6. clear - to remove all list elements
l.clear()
print(l)
# 7. using del keyword 
del list[5]
del list[5]
del list[0]
print(list)

#                   other methods 
# 8. copy - to copy list elements
m = list.copy()
print(m)
m.reverse()
print(m)