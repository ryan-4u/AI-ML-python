                                        #CREATION
str1 = 'my name is Aaryan'
str2 = "i am 20 years old"
str3 = """and i want to be a #swe"""
print(str1,str2,str3)

                                      #SOME METHODS
#1. concatenation - to add multiple strings
final_str = str1 + str2 
print(final_str)
#2. length - to get length of a string
print(len(final_str))
#3. slice method - to access part of string
print(str1[1:5])
print(str1[5:])
print(str1[:8])
print(str1[-6:])
#4. endswith() - parameter is a sub-string  - return true if string end with substring
print(str1.endswith("yan"))
print(str1.endswith("yan."))
#5. capitalize - to capital the first character
print(str1.capitalize())
#6. replace - give two parameter - (old,new)
print(str1.replace("a","x"))
#7. find - parameter is sub-string -> return -1 if not exist and first index of word if exist
print(str1.find("A"))
#8. count - to count a substring 
print(str1.count("a"))
