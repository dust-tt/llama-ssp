#!/bin/bash
export LANGUAGE="C"
export LC_ALL="C"
echo $1
count=4000
z_score=1.96
p_hat=`grep -oP "(?<=successes': )\d+" $1 | awk '{ total += $1 } END { print total}'`
p_hat=$(echo "scale=4; $p_hat / $count" | bc -l)
echo "Success rate: $p_hat"
stderr=$(echo "scale=4; sqrt($p_hat*(1-$p_hat))/sqrt($count)" | bc -l) # calculate the standard error of the sample proportion
margin=$(echo "scale=4; $z_score*$stderr" | bc -l) # calculate the margin of error
lower=$(echo "scale=4; $p_hat - $margin" | bc -l) # calculate the lower limit
upper=$(echo "scale=4; $p_hat + $margin" | bc -l) # calculate the upper limit
echo "Confidence margin: $margin"
echo "Wald confidence interval at $p probability level: ($lower, $upper)" # output the Wald confidence interval


echo -n "Confidence interval: "

