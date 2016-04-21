alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
            'r','s','t','u','v','w','x','y','z']

def check_alphabet(letter,num):
    if letter.lower() in alphabet:
        letter_low = letter.lower()
        new_index = alphabet.index(letter_low)+num
        if new_index <26:
            new_letter = alphabet[new_index]
        else:
            new_index = new_index%26
            new_letter = alphabet[new_index]
    else:
        new_letter = letter
    return new_letter

def caesar_encrypt(message,num):
    message_list = list(message)
    print message_list
    new_message = ''
    for letter in message_list:
        if letter.isupper() == True:
            new_message += check_alphabet(letter, num).upper()
            # if letter.lower() in alphabet:
            #     letter_low = letter.lower()
            #     new_index = alphabet.index(letter_low)+num
            #     if new_index <26:
            #         new_message += alphabet[new_index]
            #     else:
            #         new_index = new_index%26
            #         new_message += alphabet[new_index]
            # else:
            #     new_message += letter
        else:
            new_message +=check_alphabet(letter, num)
    return new_message

print caesar_encrypt("Caesar Cipher", 2)
print caesar_encrypt("Backwards will also work. Like this!", -13)
print caesar_encrypt("---====HeY====---", 55)
