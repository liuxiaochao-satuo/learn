from layer_naive_me import *

apple_price = 100
apple_num = 2
tax = 1.1
orange_price = 150
orange_num = 3

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_total_price = mul_apple_layer.forward(apple_price, apple_num)
orange_total_price = mul_orange_layer.forward(orange_price, orange_num)
all_total_price = add_apple_orange_layer.forward(apple_total_price, orange_total_price)
all_price = mul_tax_layer.forward(all_total_price, tax)

print(f'Total price: {all_price}')

# backward
dprice = 1
dall_total_price, dtax = mul_tax_layer.backward(dprice)
dapple_total_price, dorange_total_price = add_apple_orange_layer.backward(dall_total_price)
dorange_price, dorange_num = mul_orange_layer.backward(dorange_total_price)
dapple_price, dapple_num = mul_apple_layer.backward(dapple_total_price)

print(f"dApple price: {dapple_price} dApple num: {dapple_num}  ")
print(f"dOrange price: {dorange_price} dOrange num: {dorange_num}  ")
print(f"dTax: {dtax}")
