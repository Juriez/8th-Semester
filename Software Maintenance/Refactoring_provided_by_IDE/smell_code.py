def calculate_area_of_rectangle(length, width):
    area = length * width
    return area

def calculate_area_of_circle(radius):
    import math
    area = math.pi * (radius ** 2)
    return area

length_of_rectangle = 10
width_of_rectangle = 5
area_of_rectangle = calculate_area_of_rectangle(length_of_rectangle, width_of_rectangle)
print(f"Area of rectangle: {area_of_rectangle}")

radius_of_circle = 7
area_of_circle = calculate_area_of_circle(radius_of_circle)
print(f"Area of circle: {area_of_circle}")