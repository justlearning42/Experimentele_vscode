def get_temp(weerstand):
    #lineair:
    #source: https://www.sterlingsensors.co.uk/pt100-resistance-table
    #weerstand (ohm) -- #temp (K)
    #100 -- 273.15 -->(x2,y2)
    #18.49 -- 73.15 --> (x1, y1)
    (x1, y1) = (18.49, 73.15)
    (x2, y2) = (31.32, 103.15)
    temp = y1 + (y2-y1)/(x2-x1) * (weerstand - x1)
    rico = (y2-y1)/(x2-x1)
    return temp
#testing it with the table
print(get_temp(25.5))

(x1, y1) = (18.49, 73.15)
(x2, y2) = (31.32, 103.15)
rico = (y2-y1)/(x2-x1)
print('rico:', rico)





print('gewenste weerstand:')
print((90-y1)/rico + x1)
