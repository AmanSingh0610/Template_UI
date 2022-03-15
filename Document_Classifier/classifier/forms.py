from django import forms  
from .models import Results  


class FileForm(forms.ModelForm):
    class Meta:
        model = Results
        fields = ('filepath',)
        widgets = {
        'filepath': forms.FileInput(),
        }