# Generated by Django 3.2.15 on 2022-08-25 13:57

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='e_waste_site',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('site_id', models.CharField(max_length=10)),
                ('name', models.CharField(max_length=100)),
                ('ownership', models.CharField(max_length=10)),
                ('site', models.CharField(max_length=50)),
            ],
        ),
    ]