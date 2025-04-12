{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :inherited-members:
   :special-members: __init__, __call__
   :member-order: bysource
   :undoc-members:
   :no-index:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block parameters %}
   {% if init_params %}
   .. rubric:: {{ _('Parameters') }}

   .. autosummary::
      :toctree:
   {% for item in init_params %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block returns %}
   {% if returns %}
   .. rubric:: {{ _('Returns') }}

   .. autosummary::
      :toctree:
   {% for item in returns %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
