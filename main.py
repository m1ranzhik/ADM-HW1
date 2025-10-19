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

# Power - Mod Power
if __name__ == '__main__':
    f_number = int(input())
    power = int(input())
    s_number = int(input())

    result = pow(f_number, power)
    modulus = result - s_number * (result // s_number )

    print(result)
    print(modulus)


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
    words = line.split()
    result = "-".join(words)
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

# String Validators
if __name__ == '__main__':
    s = input()

    print(any(char.isalnum() for char in s))
    print(any(char.isalpha() for char in s))
    print(any(char.isdigit() for char in s))
    print(any(char.islower() for char in s))
    print(any(char.isupper() for char in s))

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)

# Text Wrap
import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

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
    s = s.split(' ')
    for i in range(len(s)):
        s[i] = s[i].capitalize()
    return ' '.join(s)

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
            total_earnings += price
            shoe_inventory[desired_size] -= 1

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
    np_array = numpy.array(arr, dtype=float)
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
import numpy

n, m = map(int, input().split())
lst = [list(map(int, input().split())) for i in range(n)]
numbers = numpy.array(lst)
print(numpy.mean(numbers, axis = 1))
print(numpy.var(numbers, axis = 0))
std_value = numpy.std(numbers, axis=None)
if std_value == 0:
    print(0.0)
else:
    print(f"{std_value:.11f}")

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
import numpy

n = int(input())

numbers = numpy.array([list(map(float, input().split())) for i in range(n)])
det = numpy.linalg.det(numbers)

if abs(det - round(det)) < 1e-10:
    print(f"{round(det):.1f}")
else:
    print(f"{det:.2f}")

# Set .discard(), .remove() & .pop()
def sets(numbers, lst):
    for i in range(len(lst)):
        if lst[i][0] == 'remove':
            numbers.remove(int(lst[i][1]))
        elif lst[i][0] == 'discard':(
            numbers.discard(int(lst[i][1])))
        elif lst[i][0] == 'pop':
            numbers.pop()
    return sum(numbers)

if __name__ == '__main__':
    n = int(input())
    numbers = set(map(int, input().split()))
    n_commands = int(input())
    lst = []
    for i in range(n_commands):
        command = input().split()
        lst.append(command)

    print(sets(numbers, lst))

# Check Subset
def sets(set1, set2):
    if len(set1) > len(set2):
        return False
    counter = 0
    lst1 = list(set1)
    lst2 = list(set2)
    for i in range(len(lst1)):
        if lst1[i] in lst2:
            counter += 1
    if counter == len(lst1):
        return True
    else:
        return False

if __name__ == '__main__':
    tests = int(input())
    for i in range(tests):
        a = int(input())
        set1 = set(map(int, input().split()))
        b = int(input())
        set2 = set(map(int, input().split()))
        print(sets(set1, set2))

# DefaultDict Tutorial
from collections import defaultdict

if __name__ == "__main__":
    n, m = map(int, input().split())
    d = defaultdict(list)

    # Read group A
    for i in range(1, n + 1):
        word = input().strip()
        d[word].append(i)

    # Read group B and print results
    for _ in range(m):
        word = input().strip()
        if word in d:
            print(' '.join(map(str, d[word])))
        else:
            print(-1)

# Check Strict Superset
if __name__ == '__main__':
    A = set(map(int, input().split()))
    n = int(input())
    result = True

    for _ in range(n):
        other_set = set(map(int, input().split()))
        if not (A > other_set):
            result = False
            break

    print(result)

# Collections.namedtuple()
from collections import defaultdict

if __name__ == "__main__":
    n = int(input())
    columns = input().split()
    index = 0

    for i in range(len(columns)):
        if columns[i] == 'MARKS':
            index = i

    d = defaultdict(list)
    for i in range(n):
        students = input().split()
        d[index].append(int(students[index]))

    counter = [sum(values) for key, values in d.items()]
    print(f"{counter[0] / len(d[index]):.2f}")

# Collections.OrderedDict()
from collections import defaultdict

if __name__ == "__main__":
    n = int(input())
    d = defaultdict(list)
    numbers = '0123456789'
    for i in range(n):
        item = input()
        for j in range(len(item)):
            if item[j] in numbers:
                d[item[:j]].append(int(item[j:]))
                break
    sums = {key: sum(values) for key, values in d.items()}
    for key, values in sums.items():
        print(key, values, sep='')

# Word Order
from collections import defaultdict

if __name__ == "__main__":
    n = int(input())
    d = defaultdict()
    distinct = 1
    d[input()] = 1
    for i in range(n - 1):
        word = input()
        if not word in d:
            d[word] = 1
            distinct += 1
        else:
            d[word] += 1
    print(distinct)
    for key, values in d.items():
        print(values, end=' ')

# Collections.deque()
from collections import deque

if __name__ == "__main__":
    n = int(input())
    d = deque()
    for i in range(n):
        command = input().split()
        if command[0] == 'extend':
            for j in command[1]:
                d.append(int(j))
        elif command[0] == 'extendleft':
            for j in command[1]:
                d.appendleft(int(j))
        elif command[0] == 'append':
            d.append(int(command[1]))
        elif command[0] == 'appendleft':
            d.appendleft(int(command[1]))
        elif command[0] == 'clear':
            d.clear()
        elif command[0] == 'count':
            d.count(int(command[1]))
        elif command[0] == 'pop':
            d.pop()
        elif command[0] == 'popleft':
            d.popleft()
        elif command[0] == 'remove':
            d.remove(int(command[1]))
        elif command[0] == 'reverse':
            d.reverse()
        elif command[0] == 'rotate':
            d.rotate(int(command[1]))
    print(*d)

# Piling Up!
from collections import deque

if __name__ == "__main__":
    n = int(input())
    for i in range(n):
        flag = True
        size = int(input())
        numbers = list(map(int, input().split()))
        d = deque(numbers)
        if len(d) == 1:
            print('Yes')
            continue

        if d[0] >= d[-1]:
            first = d.popleft()
        else:
            first = d.pop()
        while len(d) > 0:
            if first >= d[0] and first >= d[-1]:
                if d[0] >= d[-1]:
                    first = d.popleft()
                else:
                    first = d.pop()
            else:
                flag = False
                break
        if flag:
            print('Yes')
        else:
            print('No')

# Company Logo
import math
import os
import random
import re
import sys

if __name__ == "__main__":
    name = input()
    d = dict()
    for letter in name:
        if not letter in d:
            d[letter] = 1
        else:
            d[letter] += 1
    counter = 0
    while d and counter < 3:
        max_value = max(d.values())
        letters = sorted([key for key, values in d.items() if max_value == values])
        for i in letters:
            print(i, max_value)
            del d[i]
            counter += 1
            if counter == 3:
                break

# Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))

for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Alphabet Rangoli
def print_rangoli(size):
    alphabet = [chr(i) for i in range(97, 123)]

    lines = []
    for i in range(size):
        s = "-".join(alphabet[i:size])
        lines.append((s[::-1] + s[1:]).center(4 * size - 3, "-"))

    print('\n'.join(lines[::-1] + lines[1:]))

# The Minion Game
def minion_game(string):
    # your code goes here
    vowels = 'AEIOU'
    kevin = 0
    stuart = 0
    n = len(s)

    for i in range(n):
        if s[i] in vowels:
            kevin += n - i
        else:
            stuart += n - i

    if kevin > stuart:
        print("Kevin", kevin)
    elif stuart > kevin:
        print("Stuart", stuart)
    else:
        print("Draw")

# Merge the Tools!
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        block = string[i:i+k]
        seen = set()
        out = []
        for ch in block:
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
        print(''.join(out))

# Calendar Module
import calendar

month, day, year = map(int, input().split())
weekday = calendar.weekday(year, month, day)
day_name = calendar.day_name[weekday].upper()
print(day_name)

# Exceptions
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        a, b = input().split()
        try:
            print(int(a) // int(b))
        except ZeroDivisionError:
            print('Error Code: integer division or modulo by zero')
        except ValueError:
            if not a in '0123456789':
                print(f"Error Code: invalid literal for int() with base 10: '{a}'")
            else:
                print(f"Error Code: invalid literal for int() with base 10: '{b}'")

# Map and Lambda Function
cube = lambda x: x**3
def fibonacci(n):
    lst = []
    a, b = 0, 1
    for i in range(n):
        lst.append(a)
        a, b = b, a + b
    return lst

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# Zipped!
n, x = input().split()

lst = [map(float, input().split()) for _ in range(int(x))]
grades = list(zip(*lst))

for i in range(int(n)):
    print(f'{sum(grades[i]) / len(grades[i]):.1f}')

# Athlete Sort
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n, x = input().split()
    lst = [list(map(int, input().split())) for _ in range(int(n))]
    k = int(input())

    sorted_list = sorted(lst, key=lambda x: x[k])
    for i in range(int(n)):
        print(*sorted_list[i])

# ginortS
str = input()

upper = ''
lower = ''
odd = ''
even = ''

for symbol in str:
    if symbol.isupper():
        upper += symbol
    elif symbol.islower():
        lower += symbol
    else:
        if int(symbol) % 2 != 0:
            odd += symbol
        else:
            even += symbol

print("".join(sorted(lower)), "".join(sorted(upper)), "".join(sorted(odd)), "".join(sorted(even)), sep='')

# Recursive Digit Sum
import math
import os
import random
import re
import sys

def superDigit(n, k = 1):
    number = str(sum(list(map(int, n))) * k)
    if len(number) > 1:
        return superDigit(number)
    else:
        return number

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1]) if len(first_multiple_input) > 1 else 1
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime, timedelta

def time_delta(t1, t2):
    # Define the timestamp format (matches the problem exactly)
    fmt = "%a %d %b %Y %H:%M:%S %z"
    dt1 = datetime.strptime(t1, fmt)
    dt2 = datetime.strptime(t2, fmt)
    diff = abs(int((dt1 - dt2).total_seconds()))
    return str(diff)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for _ in range(t):
        t1 = input().strip()
        t2 = input().strip()
        fptr.write(time_delta(t1, t2) + '\n')
    fptr.close()

# Detect Floating Point Number
n = int(input())

for _ in range(n):
    number = input()
    if number != '0':
        try:
            num = float(number)
            print(True)
        except ValueError:
            print(False)
    else:
        print(False)

# Re.split()
regex_pattern = r"[.,]"

import re
print("\n".join(re.split(regex_pattern, input())))

# Group(), Groups() & Groupdict()
import re

s = input()
match = re.search(r'([a-zA-Z0-9])\1+', s)

if match:
    print(match.group(1))
else:
    print(-1)

# Re.findall() & Re.finditer()
import re

s = input().strip()
pattern = r'(?<=[qwrtypsdfghjklzxcvbnm])([aeiou]{2,})(?=[qwrtypsdfghjklzxcvbnm])'

matches = re.findall(pattern, s, flags=re.I)
print('\n'.join(matches) if matches else '-1')

# Re.start() & Re.end()
import re

S = input()
k = input()

pattern = f"(?={k})"
matches = list(re.finditer(pattern, S))

if not matches:
    print("(-1, -1)")
else:
    for m in matches:
        print(f"({m.start()}, {m.start() + len(k) - 1})")

# Regex Substitution
import re

n = int(input())
for _ in range(n):
    line = input()
    line = re.sub(r'(?<= )&&(?= )', 'and', line)
    line = re.sub(r'(?<= )\|\|(?= )', 'or', line)
    print(line)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        formatted = []
        for num in l:
            num = num[-10:]
            formatted.append(f"+91 {num[:5]} {num[5:]}")
        return f(formatted)
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

# Decorators 2 - Name Directory
import operator

def person_lister(f):
    def inner(people):
        people.sort(key=lambda x: int(x[2]))
        return [f(person) for person in people]
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# Insertion Sort - Part 1
import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    number = arr[n - 1]
    for i in range(1, n):
        if number < arr[n - 1 - i]:
            arr[n - i] = arr[n - 1 - i]
            print(*arr)
        else:
            arr[n - i] = number
            print(*arr)
            break
    else:
        arr[0] = number
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if arr[i] < arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
                i -= 1
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

# Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

import re
print(str(bool(re.match(regex_pattern, input()))))

import re

# Validating phone numbers
n = int(input())
for _ in range(n):
    number = input().strip()
    if re.fullmatch(r'[789]\d{9}', number):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
import re

n = int(input())
for _ in range(n):
    name, email_addr = input().split()
    email_addr = email_addr[1:-1]
    if re.match(r'^[a-zA-Z][\w\.-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$', email_addr):
        print(f"{name} <{email_addr}>")

# Hex Color Code
import re

n = int(input())
inside = False
pat = re.compile(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?(?=[;,\s\)])')

for _ in range(n):
    line = input()
    if '{' in line:
        inside = True
        continue
    if '}' in line:
        inside = False
        continue
    if inside:
        for m in pat.findall(line):
            print(m)

# HTML Parser - Part 1
from html.parser import HTMLParser
import sys

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for k, v in attrs:
            print(f"-> {k} > {v if v is not None else 'None'}")

    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for k, v in attrs:
            print(f"-> {k} > {v if v is not None else 'None'}")

    def handle_endtag(self, tag):
        print(f"End   : {tag}")

n = int(sys.stdin.readline())
html = "".join(sys.stdin.readline() for _ in range(n))

parser = MyHTMLParser()
parser.feed(html)
parser.close()

# HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)

html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
import sys

class TagAttrParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for k, v in attrs:
            print(f"-> {k} > {v}")

    def handle_startendtag(self, tag, attrs):
        print(tag)
        for k, v in attrs:
            print(f"-> {k} > {v}")

    def handle_comment(self, data):
        pass
    def handle_data(self, data):
        pass

n = int(sys.stdin.readline())
html = "".join(sys.stdin.readline() for _ in range(n))

parser = TagAttrParser()
parser.feed(html)
parser.close()

# Validating UID
import re

for _ in range(int(input())):
    uid = input().strip()

    if (len(uid) == 10 and uid.isalnum() and len(set(uid)) == 10 and len(re.findall(r'[A-Z]', uid)) >= 2 and len(re.findall(r'\d', uid)) >= 3):
        print("Valid")
    else:
        print("Invalid")

# Birthday Cake Candles
import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    maximum = max(candles)
    result = 0
    for i in range(len(candles)):
        if maximum == candles[i]:
            result += 1
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Viral Advertising
import math
import os
import random
import re
import sys

def viralAdvertising(n):
    shared = 5
    cumulative = 0
    for i in range(n):
        cumulative += math.floor(shared / 2)
        shared = math.floor(shared / 2) * 3
    return cumulative
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if v1 <= v2:
        return "NO"
    return "YES" if (x2 - x1) % (v1 - v2) == 0 else "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    total = len(node.attrib)
    for child in node:
        total += get_attr_number(child)
    return total

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if level == maxdepth:
        maxdepth += 1
    for child in elem:
        depth(child, level + 1)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
