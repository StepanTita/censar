# Create your views here.
from urllib.parse import urlencode

from django.shortcuts import render, redirect
from django.urls import reverse, reverse_lazy
from django.views.generic import TemplateView, RedirectView, FormView, View
from django.core.files.storage import default_storage

import censar
from censar import settings
from process.process import get_processed
from text_recognition import forms


class IndexView(TemplateView):
    template_name = 'wrapper.html'

    def get_context_data(self, **kwargs):
        context = super(TemplateView, self).get_context_data(**kwargs)
        return context


class InitialView(RedirectView):
    permanent = False
    query_string = False
    pattern_name = 'index'


class ImagesView(FormView):
    template_name = 'contact.html'
    form_class = forms.ImagesForm
    success_url = reverse_lazy('result')

    def form_valid(self, form):
        # process image here
        return super().form_valid(form)

    def post(self, request, *args, **kwargs):
        file = request.FILES['images']
        file_name = default_storage.save(file.name, file)

        base_url = self.get_success_url()  # 1 /products/
        query_string = urlencode({'image': file_name})  # 2 category=42
        url = '{}?{}'.format(base_url, query_string)  # 3 /products/?category=42
        return redirect(url)


class ResultView(View):
    template_name = 'result.html'
    success_url = reverse_lazy('result')

    def get(self, request, *args, **kwargs):
        return render(request, context={'image': get_processed(request.GET.get('image'))}, template_name=self.template_name)
