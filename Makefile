minor:
	bumpversion minor --allow-dirty
	git push
	git push --tags
patch:
	bumpversion patch --allow-dirty
	git push
	git push --tags

docs.deploy:
	 mkdocs gh-deploy

docs.serve:
	mkdocs serve