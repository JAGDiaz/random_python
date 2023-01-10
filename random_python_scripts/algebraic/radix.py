
def base_str_to_int(s, base=2):
    return sum(int(s[ii])*base**(len(s) - ii-1) for ii in range(len(s)))

def to_dec(number, base=2):
    number = number[::-1]
    floating = number.rfind('.')
    number = number[:floating] + number[floating+1:]
    return sum(int(digit)*base**exp for (exp, digit) in zip(range(-floating, len(number)-floating), number))


def base_change(num, new_base=16):
    symbols = list('0123456789ABCDEF')
    new_num = '' if num != 0 else '0'
    while num > 0:
        new_num += symbols[num%new_base]
        num //= new_base
    return new_num[::-1]

def base_change_frac(num, new_base=16):
    new_num = '.' if num != 0 else '.0'
    while num > 0:
        temp = num*new_base
        new_num += f"{int(temp)}"
        num = temp - int(temp)
    return new_num



def base_change_float(num, old_base=None, new_base=16):
    if isinstance(num, str):
        num = to_dec(num, old_base)

    whole, part = int(num), num - int(num)
    whole = base_change(whole, new_base=new_base)
    part = base_change_frac(part, new_base=new_base)
    return whole + part

if __name__ == '__main__':
    string = "10010"

    print(base_str_to_int(string)) 

    print(f"{0b10010:d}")

    num = 7

    print(base_change_float(9.75, new_base=2))

    print(to_dec("234.54", 16))
    
    


    