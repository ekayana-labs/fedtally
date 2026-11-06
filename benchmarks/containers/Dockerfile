FROM rust:1-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake clang pkg-config
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p fedtally-cli

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends iproute2
COPY --from=build /src/target/release/fedtally /usr/local/bin/fedtally
COPY benchmarks/containers/setup.py benchmarks/containers/train.py /opt/
