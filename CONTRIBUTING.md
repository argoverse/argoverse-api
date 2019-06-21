# Contributing to Argoverse

Contributions to Argoverse are welcomed!  If you find a bug please open an issue on the github page describing the problem with steps to reproduce.  Better yet, if you figure out a fix, please open a pull request!

To open a pull request here are some steps to get you started:

- [Fork the Argoverse repository](https://help.github.com/en/articles/fork-a-repo) and [clone to your machine](https://help.github.com/en/articles/cloning-a-repository).

- Create a branch for your changes.
  - `$ git checkout -b <name of your branch>`

- [Install pre-commit](https://pre-commit.com/#install)
- Run `pre-commit install --install-hooks` to install the [git hooks](https://githooks.com/) for this repo, so your changes get auto-formatted as you commit.

- Validate that your changes do not break any existing unit tests.
  - Run all unit tests: `$ pytest tests`

- Please provide documentation for any new code your pull request provides.

- Push your branch to the remote when it is complete.
  - `$ git push --set-upstream origin <name of your branch>`

- [Open a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork) https://github.com/argoai/argoverse-api/pulls from your branch.
  - Hint: having unit tests that validate your changes with your pull
    request will help to land your changes faster.


## Helpful documentation

Making a contribution to Argoverse will require some basic knowledge of Git and Github.  If these tools are new to you, here are is some documentation that might be helpful.

- Basic tutorial for using Git and Github: https://guides.github.com/activities/hello-world/
- Forking a repository: https://help.github.com/en/articles/fork-a-repo
- Examples for formatting documentation: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google
- Creating a pull request from a fork: https://help.github.com/en/articles/creating-a-pull-request-from-a-fork

## License

By contributing to this project you are acknowledging and agreeing that all contributions made by you are submitted under the terms of the [MIT license](https://opensource.org/licenses/MIT).
