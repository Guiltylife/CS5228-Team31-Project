# CS5228-Team31-Project

**When uploading a new version, please describe the changes in the new version in the ReadMe file.**

## Version Changes

### Version 1:

Encoding part:

1.Change the encoding of rent_approval_date and lease commence date from int to date type.

2.Improve the binary encoding. The binary encoding result is splited into columns, and each column contains data of type int.

3.For town, block, street_name, flat_model, subzone, planning_area and region, I use binary encoding instead of dropping. If they don't contribute to the result, you can drop them in prediction part.

4.For latitude, longitude and monthly_rent, I normalized them but also kept the original values. You can choose them based on the predicted effect. **Important: You can only choose either monthly_rent or monthly_rent_norm in your training and predicting.**