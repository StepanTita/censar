from django import forms


class ImagesForm(forms.Form):
    images = forms.ImageField(
        widget=forms.FileInput(attrs={'class': 'file-upload-input', 'onchange': 'readURL(this);', 'accept': 'image/*'}))
