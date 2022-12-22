from django.db import models


# Create your models here.
class Council(models.Model):
    gov_area = models.CharField(max_length=128)

    def __str__(self):
        return str(self.gov_area)


class Centre_Type(models.Model):
    type = models.CharField(max_length=128)

    def __str__(self):
        return str(self.type)


class EWasteSite(models.Model):
    name = models.CharField(max_length=128)
    ownership = models.ForeignKey(Centre_Type, on_delete=models.CASCADE)
    site = models.ForeignKey(Council, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.name)

class Suburb(models.Model):
    district = models.CharField(max_length=128)

    def __str__(self):
        return str(self.district)

class Clothing_Disp_Centre(models.Model):
    name = models.CharField(max_length=128)

    def __str__(self):
        return str(self.name)

class Clothing(models.Model):
    name = models.ForeignKey(Clothing_Disp_Centre,on_delete=models.CASCADE)
    address = models.CharField(max_length =500)
    district = models.ForeignKey(Suburb,on_delete=models.CASCADE)

    def __str__(self):
        return str(self.address)
