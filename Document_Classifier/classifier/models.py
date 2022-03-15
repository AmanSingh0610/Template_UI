from django.db import models

# Create your models here.
class Results(models.Model):
    docname = models.CharField(db_column='Document Name', max_length=100)  
    confidence = models.IntegerField(db_column='Confidence Score')
    ddt = models.BooleanField(db_column='DDT Classification?')
    keywords = models.TextField(db_column='Keywords Identified')
    filepath= models.FileField(db_column='Document',upload_to='files/', null=True, verbose_name="") 

    class Meta:
        db_table = 'results'

    def __str__(self):
        return self.docname + ": " + str(self.filepath)