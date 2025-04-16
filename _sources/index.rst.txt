:html_theme.sidebar_secondary.remove:

hpiPy: House Price Indices in Python
====================================

.. role:: raw-html(raw)
   :format: html

**Version**: |version|

.. rst-class:: lead

   **hpiPy** simplifies and standardizes the creation of house price indices in Python.

   The package provides tools to evaluate index quality through predictive accuracy, volatility, and revision metrics—enabling meaningful comparisons across different methods and estimators. It focuses on the most widely used approaches: repeat sales and hedonic pricing models, with support for base, robust, and weighted estimators where applicable. It also includes a random forest–based method paired with partial dependence plots to derive index components, as well as a neural network approach that separates property-specific and market-level effects to jointly estimate quality and index components from property-level data. Based on `hpiR <https://github.com/andykrause/hpiR>`_.

.. grid:: 1 1 2 2
   :padding: 0 2 3 5
   :gutter: 2 2 3 3
   :class-container: startpage-grid

   .. grid-item-card:: Installation
      :link: installation
      :link-type: ref
      :link-alt: Installation

      Installation instructions and quick setup guide to help you start creating house price indices.

   .. grid-item-card:: Quick Start
      :link: quickstart
      :link-type: ref
      :link-alt: Quick Start

      Practical examples demonstrating how to create and evaluate house price indices.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: ref
      :link-alt: User guide

      Comprehensive documentation on all supported methods, from repeat sales to machine learning.

   .. grid-item-card:: API
      :link: api
      :link-type: ref
      :link-alt: API

      Information on all of the package methods and classes, for when you want just the details.

.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <installation/installation>
   Quick Start <quick_start/quick_start>
   User Guide <user_guide/index>
   API <api/index>
   Release Notes <releases/changes>

.. _GitHub: https://github.com/reidjohnson/hpipy
