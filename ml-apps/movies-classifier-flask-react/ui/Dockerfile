# Use an official node runtime as a parent image
FROM node:10.4.1

WORKDIR /app/

# Install dependencies
COPY package.json yarn.lock /app/

RUN npm install

# Add rest of the client code
COPY . /app/

EXPOSE 3000

# start app
CMD ["npm", "start"]