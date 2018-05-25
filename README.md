# KTH-DD2424-Assignment4

In this assignment we will train an RNN to synthesize English text character by character. We will train a *vanilla RNN* using the text from the book *The Goblet of Fire* by J.K. Rowling.

## Read in the data

In this part we read in the text file and get all unique characters. Then we get two dictionaries `char_to_ind` and `ind_to_char` seperately.(You can print the `char` and can find that Rowling does not use number 5 and 8 in the whole book.)

## Initialize parameters and synthesize text from randomly initialized RNN

In this part we initialized the parameters for RNN and write a function to synthesize text from the randomly initialized RNN.
