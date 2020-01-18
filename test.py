import kNN
testVector = kNN.img2vector('testDigits/0_13.txt')
print(testVector[0, 0:31])
kNN.hanwritingClassTest()
