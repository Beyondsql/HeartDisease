denominations = [(.01, "penny"),
                 (.05, "nickel"),
                 (.10, "dime"),
                 (10, "ten dollar bill"),
                 (.25, "quarter"),
                 (20, "twenty dollar bill"),
                 (1, "dollar bill"),
                 (5, "five dollar bill")

                 ]

for money in denominations:
    print money[0]
    print 21.01-money[0]

print sorted(denominations, key=lambda x:x[0], reverse=True)

def calculateChange(cost, paid_amount):
    from collections import defaultdict
    change_amt = paid_amount - cost
    if change_amt < 0:
        return "give more money"
    print change_amt
    new_denom = sorted(denominations, key=lambda x: x[0], reverse=True)
    #denominations.reverse()
    print new_denom
    change_given = defaultdict(int)
    while change_amt > 0.001:
        print "while loop started"
        print "change amount is currently" + str(change_amt)
        for money in new_denom:
            print money
            print change_amt>money[0]
            print change_amt-money[0]
            if change_amt+.0001>=money[0]:
                print "proper denomination found"
                change_amt = change_amt-money[0]
                print "new change amount is" + str(change_amt)
                change_given[money[1]] += 1
                break
    return change_given

print calculateChange(77.73, 80.00)
print calculateChange(21,20)
