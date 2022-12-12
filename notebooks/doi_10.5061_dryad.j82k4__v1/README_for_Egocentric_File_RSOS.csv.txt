The file Egocentric_File_RSOS.csv contains data corresponding to egocentric networks of 3,106,293 individuals derived from anonymized call detail records for a single month in the year 2007 from a mobile phone service provider in a European country. Each line in the file corresponds to a different ego-alter pair and provides aggregated voice calling information for the concerned month.


The format (each line) of the data is the following:

Ego no., Ego's gender, Ego's age, Alter's gender, Alter's age, Total no. of calls, Total time (secs), Total number of days of communication


Flags used to denote the gender: 1=female, 2=male

In case the gender or age information is not available the field is substituted using a '0'.


For example, the following line

1,1,46,2,58,5,892,4


denotes:


The ego numbered 1 is a female of age 46, has a male alter of age 58 and that the pair has participated in 5 calls amounting 892 seconds on 4 different days in the month.

