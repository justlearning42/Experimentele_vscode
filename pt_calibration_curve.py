def get_temp(weerstand):
    #lineair:
    #source: https://www.sterlingsensors.co.uk/pt100-resistance-table
    #weerstand (ohm) -- #temp (K)
    #100 -- 273.15 -->(x2,y2)
    #18.49 -- 73.15 --> (x1, y1)
    (x1, y1) = (18.49, 73.15)
    (x2, y2) = (31.32, 103.15)
    temp = y1 + (y2-y1)/(x2-x1) * (weerstand - x1)
    return temp
#testing it with the table
print(get_temp(100) - 273.15) #0
print(get_temp(18.49) - 273.15) #-200
print(get_temp(59.04)-273.15) #-103
print(get_temp(40.96)-273.15) #-147

