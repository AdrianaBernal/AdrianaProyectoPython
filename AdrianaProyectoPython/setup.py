from setuptools import setup, find_packages

setup(
   name='AdrianaProyectoPython',
   version='0.0.1',
   author='Adriana Bernal',
   author_email='abernal018@ikasle.ehu.eus',
   packages=find_packages(),
   url='https://github.com/tu_usuario/mi_paquete',
   license='LICENSE.txt',
   description='Paquete para preprocesamiento, análisis y visualización de datasets',
   long_description=open('README.txt').read(),
   tests_require=['pytest'],
   install_requires=[
      "seaborn >= 0.9.0",
      "pandas >= 0.25.1",
      "matplotlib >= 3.1.1",
      "numpy >=1.17.2"
   ],
)