from django.shortcuts import render
from django.views.generic import TemplateView
from service.emojifiers import api

class Index(TemplateView):
    template_name='index.html'

    def post(self,request):
        content=request.POST['content']
        emoji=api.predict(content)

        content={
            'content': content,
            'emoji': emoji


        }
        return render(request,self.template_name,content)
# Create your views here.