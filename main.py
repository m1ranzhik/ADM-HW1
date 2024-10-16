# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else
if __name__ == '__main__':
    n = int(input().strip())
if n % 2 != 0:
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:
    print("Weird")
elif n % 2 == 0 and n > 20:
    print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a // b)
    print(a / b)

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(0,n):
        print(i*i)

# Write a function
def is_leap(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

year = int(input())
print(is_leap(year))

# Print function
if __name__ == '__main__':
    n = int(input())
i = 1
while i <= n and 1 <= n <= 150:
        print(i, end="")
        i += 1

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
coordinates = []

for i in range(x + 1):
    for j in range(y + 1):
        for k in range(z + 1):
            if i + j + k != n:
                coordinates.append([i, j, k])
print(coordinates)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    set_arr = set(arr)
    arr2 = sorted(list(set_arr))
    print(arr2[-2])

# Nested Lists
if __name__ == '__main__':
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])
    grades = sorted(set([score for name, score in students]))
    second_lowest_grade = grades[1]
    second_lowest_students = [name for name, score in students if score == second_lowest_grade]

    second_lowest_students.sort()
    for student in second_lowest_students:
        print(student)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    average_marks = sum(student_marks[query_name]) / len(student_marks[query_name])

    print(f"{average_marks:.2f}")

# Lists
if __name__ == '__main__':
    N = int(input())
    list = []

    for _ in range(N):
        command = input().split()
        operation = command[0]

        if operation == 'insert':
            index = int(command[1])
            number = int(command[2])
            list.insert(index, number)

        elif operation == 'print':
            print(list)

        elif operation == 'remove':
            number = int(command[1])
            list.remove(number)

        elif operation == 'append':
            number = int(command[1])
            list.append(number)

        elif operation == 'sort':
            list.sort()

        elif operation == 'pop':
            list.pop()

        elif operation == 'reverse':
            list.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    tup = tuple(map(int, input().split()))
    print(hash(tup))

# sWAP cASE
def swap_case(s):
    result = ''
    for letter in s:
        if letter.islower():
            result += letter.upper()
        elif letter.isupper():
            result += letter.lower()
        else:
            result += letter
    return result

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# String Split and Join
def split_and_join(line):
    words = line.split()  # Split the string by spaces
    result = "-".join(words)  # Join the words with a hyphen '-'
    return result

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find a String
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i + len(sub_string)] == sub_string:
            count += 1
    return count


if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)

# Text Wrap
import textwrap

def wrap(string, max_width):
    return

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat
def print_door_mat(n, m):
    for i in range(1, n, 2):
        str = '.|.' * i
        print(str.center(m,'-'))

    print('WELCOME'.center(m, '-'))

    for i in range(n - 2, 0, -2):
        str = '.|.' * i
        print(str.center(m, '-'))

n, m = map(int, input().split())

print_door_mat(n, m)

# String Formatting
def print_formatted(n):
    width = len(bin(n)[2:])

    for i in range(1, n + 1):
        print(
            f"{str(i).rjust(width)} {oct(i)[2:].rjust(width)} {hex(i)[2:].upper().rjust(width)} {bin(i)[2:].rjust(width)}")


if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# Capitalize!
import math
import os
import random
import re
import sys

def solve(s):

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = solve(s)
    fptr.write(result + '\n')
    fptr.close()

# Introduction to Sets
def average(array):
    array = set(arr)
    avg = sum(array) / len(array)
    return round(avg, 3)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# Symmetric Difference
def symmetric_difference(set_a, set_b):
    sym_diff = set_a.symmetric_difference(set_b)
    sorted_diff = sorted(sym_diff)

    for number in sorted_diff:
        print(number)

if __name__ == '__main__':
    m = int(input())
    set_a = set(map(int, input().split()))
    n = int(input())
    set_b = set(map(int, input().split()))

    symmetric_difference(set_a, set_b)

# No Idea!
def calculate_happiness(array, liked_set, disliked_set):
    happiness = 0
    # Iterate through each number in the array
    for number in array:
        if number in liked_set:
            happiness += 1
        elif number in disliked_set:
            happiness -= 1

    return happiness  

if __name__ == '__main__':
    n, m = map(int, input().split())
    array = list(map(int, input().split()))
    liked_set = set(map(int, input().split()))
    disliked_set = set(map(int, input().split()))

    final_happiness = calculate_happiness(array, liked_set, disliked_set)
    print(final_happiness)

# Set .add()
if __name__ == '__main__':
    n = int(input())
    distinct_stamps = set()

    for _ in range(n):
        country = input()
        distinct_stamps.add(country)

    print(len(distinct_stamps))

# Set .union() Operation
if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    m = int(input())
    french_subscribers = set(map(int, input().split()))

    # Find the union of both sets to get all unique students
    all_subscribers = english_subscribers.union(french_subscribers)

    print(len(all_subscribers))

# Set .intersection() Operation
if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    m = int(input())
    french_subscribers = set(map(int, input().split()))

    all_subscribers = english_subscribers.intersection(french_subscribers)

    print(len(all_subscribers))

# Set .difference() Operation
if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    m = int(input())
    french_subscribers = set(map(int, input().split()))

    all_subscribers = english_subscribers.difference(french_subscribers)

    print(len(all_subscribers))

# Set .symmetric_difference() Operation
if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    m = int(input())
    french_subscribers = set(map(int, input().split()))

    all_subscribers = english_subscribers.symmetric_difference()(french_subscribers)

    print(len(all_subscribers))

# Set Mutations
if __name__ == '__main__':
    n = int(input())
    initial_set = set(map(int, input().split()))
    m = int(input())

    for _ in range(m):
        operation_info = input().split()
        operation_name = operation_info[0]
        other_set = set(map(int, input().split()))  # Create the other set

        if operation_name == 'intersection_update':
            initial_set.intersection_update(other_set)
        elif operation_name == 'update':
            initial_set.update(other_set)
        elif operation_name == 'symmetric_difference_update':
            initial_set.symmetric_difference_update(other_set)
        elif operation_name == 'difference_update':
            initial_set.difference_update(other_set)

    print(sum(initial_set))

# The Captain's Room
if __name__ == '__main__':
    group_size = int(input())
    room_numbers = list(map(int, input().split()))
    room_count = {}

    for room in room_numbers:
        if room in room_count:
            room_count[room] += 1
        else:
            room_count[room] = 1

    for room, count in room_count.items():
        if count == 1:
            captain_room = room
            break

    print(captain_room)

# collections.Counter()
from collections import Counter

if __name__ == '__main__':
    n = int(input())
    shoe_sizes = list(map(int, input().split()))

    shoe_inventory = Counter(shoe_sizes)
    
    m = int(input())

    total_earnings = 0
    for _ in range(m):
        desired_size, price = map(int, input().split())
        if shoe_inventory[desired_size] > 0:
            total_earnings += price  # Add the price to total earnings
            shoe_inventory[desired_size] -= 1  # Decrease the available stock

    print(total_earnings)

# Exceptions
def main():
    test_cases = int(input())

    for _ in range(test_cases):
        values = input().split()
        a, b = values

        try:
            result = int(a) // int(b)
            print(result)
        except ZeroDivisionError as e:
            print("Error Code:", e)
        except ValueError as e:
            print("Error Code:", e)


if __name__ == "__main__":
    main()

# Zipped!
def main():
    n, m = map(int, input().split())

    marks = []

    for _ in range(m):
        subject_marks = list(map(float, input().split()))
        marks.append(subject_marks)

    averages = [sum(student_marks) / m for student_marks in zip(*marks)]

    for avg in averages:
        print(f"{avg:.1f}")


if __name__ == "__main__":
    main()

#Arrays 
import numpy

def arrays(arr):
    # Convert the input list to a NumPy array with float type
    np_array = numpy.array(arr, dtype=float)
    # Reverse the NumPy array
    reversed_array = np_array[::-1]
    return reversed_array
arr = input().strip().split(' ')
result = arrays(arr)
print(result)
# Transpose and Flatten 
import numpy as np

def transpose_and_flatten():
    n, m = map(int, input().strip().split())
    a = np.array([input().strip().split() for _ in range(n)], dtype=int)
    
    trans = a.T
    flat = a.flatten()
    
    print(trans)
    print(flat)

transpose_and_flatten()

# Shape and Reshape
import numpy
arr = list(map(int, input().split()))
my_array = numpy.reshape(arr,(3,3))
print(my_array)

# Concatenate 
import numpy as np

n, m, p = map(int, input().split())

array1 = np.array([input().split() for _ in range(n)], int)
array2 = np.array([input().split() for _ in range(m)], int)

result = np.concatenate((array1, array2), axis=0)
print(result)

# Zeros and Ones
import numpy as np

shape = tuple(map(int, input().split()))

zeros_array = np.zeros(shape, dtype=int)
print(zeros_array)

ones_array = np.ones(shape, dtype=int)
print(ones_array)

# Eye and Identity 
import numpy as np
np.set_printoptions(legacy='1.13')  
n, m = map(int, input().split())

print(np.eye(n, m))

# Array Mathematics 
import numpy as np

n, m = map(int, input().split())

A = np.array([input().split() for _ in range(n)], dtype=int)
B = np.array([input().split() for _ in range(n)], dtype=int)

print(A + B)       
print(A - B)          
print(A * B)         
print(A // B)        
print(A % B)         
print(A ** B)       

# Floor, Ceil and Rint
import numpy as np

np.set_printoptions(legacy='1.13')
a = np.array(input().split(), dtype=float)

fl = np.floor(a)
cl = np.ceil(a)
rin = np.rint(a)

print(fl, cl, rin, sep = '\n')

# Sum and Prod
import numpy as np

n, m = map(int, input().split())
a = np.array([input().split() for _ in range(n)], dtype=int)

summary = np.sum(a, axis = 0)
product = np.prod(summary)

print(product)

# Min and Max
import numpy as np
n, m = map(int, input().split())  
a = np.array([list(map(int, input().split())) for _ in range(n)])

minimum = np.min(a, axis = 1)
maximum = np.max(minimum)

print(maximum)

# Mean, Var and Std
import numpy as np

n, m = map(int, input().split())
a = np.array([input().split() for _ in range (n)], dtype = int)

first = np.mean(a, axis = 1)
second = np.var(a, axis = 0)
third = np.std(a, axis = None)

print(first)
print(second)
print(f"{third:.11f}")

# Dot and Cross 
import numpy as np

n = int(input())
a = np.array([input().split() for _ in range (n)], dtype = int)
b = np.array([input().split() for _ in range (n)], dtype = int)

print(np.dot(a, b))

# Inner and Outer 
import numpy as np

a = np.array(input().split(), dtype= int)
b = np.array(input().split(), dtype= int)

print(np.inner(a, b))
print(np.outer(a, b))

# Polynomials 
import numpy as np
a = list(map(float, input().split()))
n = float(input())

print(np.polyval(a, n))

# Linear Algebra
import numpy as np
n = int(input())
a = np.array([input().split() for _ in range(n)], dtype = float)

det_a = np.linalg.det(a)
print(round(det_a, 2))







